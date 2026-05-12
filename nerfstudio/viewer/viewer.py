# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manage the state of the viewer"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import requests
import numpy as np
import torch
import viser
import viser.theme
import viser.transforms as vtf
from typing_extensions import assert_never
from flask import Flask, Response, jsonify, stream_with_context
from flask_cors import CORS

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName
from nerfstudio.viewer.control_panel import ControlPanel
from nerfstudio.viewer.export_panel import populate_export_tab
from nerfstudio.viewer.render_panel import populate_render_tab
from nerfstudio.viewer.render_state_machine import RenderAction, RenderStateMachine
from nerfstudio.viewer.utils import CameraState, parse_object
from nerfstudio.viewer.viewer_elements import ViewerControl, ViewerElement, ViewerVec3
from nerfstudio.viewer_legacy.server import viewer_utils

if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer


VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0


@decorate_all([check_main_thread])
class Viewer:
    """Class to hold state for viewer variables

    Args:
        config: viewer setup configuration
        log_filename: filename to log viewer output to
        datapath: path to data
        pipeline: pipeline object to use
        trainer: trainer object to use
        share: print a shareable URL

    Attributes:
        viewer_info: information string for the viewer
        viser_server: the viser server
    """

    viewer_info: List[str]
    viser_server: viser.ViserServer

    def __init__(
        self,
        config: cfg.ViewerConfig,
        log_filename: Path,
        datapath: Path,
        pipeline: Pipeline,
        pipeline_b: Optional[Pipeline] = None,
        experiment_name_a: Optional[str] = None,
        experiment_name_b: Optional[str] = None,
        trainer: Optional[Trainer] = None,
        train_lock: Optional[threading.Lock] = None,
        share: bool = False,
    ):
        self.ready = False  # Set to True at end of constructor.
        self.config = config
        self.trainer = trainer
        self.last_step = 0
        self.train_lock = train_lock
        self.pipeline = pipeline
        self.pipeline_b = pipeline_b
        self.active_pipeline_idx = 0  # 0 = pipeline, 1 = pipeline_b
        self.experiment_name_a = experiment_name_a if experiment_name_a else "Model A"
        self.experiment_name_b = experiment_name_b if experiment_name_b else "Model B"
        self.compare_trans = None
        self.compare_rot = None
        self.compare_reset = None
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath
        self.include_time = self.pipeline.datamanager.includes_time

        if self.config.websocket_port is None:
            websocket_port = viewer_utils.get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.log_filename.parent.mkdir(exist_ok=True)

        # viewer specific variables
        self.output_type_changed = True
        self.output_split_type_changed = True
        self.step = 0
        self.train_btn_state: Literal["training", "paused", "completed"] = (
            "training" if self.trainer is None else self.trainer.training_state
        )
        self._prev_train_state: Literal["training", "paused", "completed"] = self.train_btn_state
        self.last_move_time = 0
        # track the camera index that last being clicked
        self.current_camera_idx = 0

        self.viser_server = viser.ViserServer(host=config.websocket_host, port=websocket_port)
        # Set the name of the URL either to the share link if available, or the localhost
        share_url = None
        if share:
            share_url = self.viser_server.request_share_url()
            if share_url is None:
                print("Couldn't make share URL!")

        if share_url is not None:
            self.viewer_info = [f"Viewer at: http://localhost:{websocket_port} or {share_url}"]
        elif config.websocket_host == "0.0.0.0":
            # 0.0.0.0 is not a real IP address and was confusing people, so
            # we'll just print localhost instead. There are some security
            # (and IPv6 compatibility) implications here though, so we should
            # note that the server is bound to 0.0.0.0!
            self.viewer_info = [f"Viewer running locally at: http://localhost:{websocket_port} (listening on 0.0.0.0)"]
        else:
            self.viewer_info = [f"Viewer running locally at: http://{config.websocket_host}:{websocket_port}"]

        # Initialize telemetry HTTP server
        self.telemetry_port = websocket_port + 1000
        # Harvest integration env vars (set by SLURM viewer script)
        self.harvest_api_url = os.environ.get("HARVEST_API_URL", "")
        self.harvest_api_token = os.environ.get("HARVEST_API_TOKEN", "")
        self.harvest_region_model_id = os.environ.get("HARVEST_REGION_MODEL_ID", "")
        self.harvest_region_model_id_b = os.environ.get("HARVEST_REGION_MODEL_ID_B", "")
        self.harvest_config_path = os.environ.get("HARVEST_CONFIG_PATH", "")
        self.waypoint_handles: Dict[str, list] = {}  # safe_name → [sphere_handle, label_handle]
        self.measurement_handles: list = []
        self.telemetry_app = Flask("nerfstudio_telemetry")
        CORS(self.telemetry_app)  # Allow cross-origin requests from Harvest

        @self.telemetry_app.route("/telemetry", methods=["GET"])
        def get_telemetry():
            clients_data = []
            clients = self.viser_server.get_clients()
            for client_id, client in clients.items():
                clients_data.append({
                    "client_id": client_id,
                    "position": client.camera.position.tolist(),
                    "wxyz": client.camera.wxyz.tolist(),
                    "fov": float(client.camera.fov),
                    "aspect": float(client.camera.aspect),
                })
            return jsonify({"clients": clients_data})

        @self.telemetry_app.route("/teleport", methods=["POST"])
        def teleport_camera():
            from flask import request
            data = request.get_json()

            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            position = data.get("position")
            wxyz = data.get("wxyz")
            fov = data.get("fov")
            client_id = data.get("client_id", 0)  # Default to first client

            if position is None or wxyz is None:
                return jsonify({"error": "Missing position or wxyz"}), 400

            clients = self.viser_server.get_clients()
            if client_id not in clients:
                return jsonify({"error": f"Client {client_id} not found"}), 404

            client = clients[client_id]

            # Set camera with atomic update
            with client.atomic():
                client.camera.position = tuple(position)
                client.camera.wxyz = tuple(wxyz)
                if fov is not None:
                    client.camera.fov = float(fov)

            return jsonify({"success": True, "client_id": client_id})

        @self.telemetry_app.route("/scene_bounds", methods=["GET"])
        def get_scene_bounds():
            try:
                scene_box = self.pipeline.datamanager.train_dataset.scene_box
                aabb = scene_box.aabb.cpu().numpy()

                min_point = aabb[0].tolist()
                max_point = aabb[1].tolist()
                center = scene_box.get_center().cpu().numpy().tolist()
                diagonal = float(scene_box.get_diagonal_length())

                return jsonify({
                    "min": min_point,
                    "max": max_point,
                    "center": center,
                    "diagonal": diagonal
                })
            except AttributeError:
                return jsonify({"error": "Scene box not available"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.telemetry_app.route("/nearest_views", methods=["GET"])
        def get_nearest_views():
            from flask import request
            import json

            try:
                position = request.args.get("position")
                wxyz = request.args.get("wxyz")
                n_param = request.args.get("n")
                max_distance_param = request.args.get("max_distance")

                if not position or not wxyz:
                    return jsonify({"error": "Missing position or wxyz"}), 400

                pos = json.loads(position)
                wxyz_val = json.loads(wxyz)

                if len(pos) != 3 or len(wxyz_val) != 4:
                    return jsonify({"error": "Invalid dimensions"}), 400

                viewer_pos_ns = np.array(pos) / VISER_NERFSTUDIO_SCALE_RATIO

                R = vtf.SO3(wxyz=np.array(wxyz_val))
                R = R @ vtf.SO3.from_x_radians(np.pi)
                viewer_dir = -R.as_matrix()[:, 2]

                active_pipeline = self.pipeline_b if self.active_pipeline_idx == 1 and self.pipeline_b else self.pipeline
                cameras = active_pipeline.datamanager.train_dataset.cameras
                candidates = []

                for idx in range(len(cameras)):
                    c2w = cameras.camera_to_worlds[idx]
                    cam_pos = c2w[:3, 3].cpu().numpy()
                    cam_dir = -c2w[:3, 2].cpu().numpy()
                    dist = float(np.linalg.norm(viewer_pos_ns - cam_pos))
                    dot = np.dot(viewer_dir, cam_dir)
                    if dot > 0:
                        candidates.append((idx, dist))

                if not candidates:
                    candidates = [
                        (i, float(np.linalg.norm(viewer_pos_ns - cameras.camera_to_worlds[i][:3, 3].cpu().numpy())))
                        for i in range(len(cameras))
                    ]

                if not candidates:
                    return jsonify({"error": "No cameras available"}), 404

                candidates.sort(key=lambda x: x[1])

                from pathlib import Path
                dataparser_outputs = active_pipeline.datamanager.train_dataset._dataparser_outputs
                dataparser_scale = dataparser_outputs.dataparser_scale

                if max_distance_param:
                    # max_distance_param is in meters — convert to scene units for filtering
                    max_dist_scene = float(max_distance_param) * dataparser_scale if dataparser_scale else float(max_distance_param)
                    candidates = [(idx, dist) for idx, dist in candidates if dist <= max_dist_scene]
                else:
                    candidates = candidates[:int(n_param) if n_param else 1]

                pipeline_idx = 1 if (self.active_pipeline_idx == 1 and self.pipeline_b) else 0
                image_filenames = dataparser_outputs.image_filenames
                results = [
                    {
                        "index": int(idx),
                        "filename": Path(image_filenames[idx]).name,
                        "distance": dist,
                        "distance_meters": dist / dataparser_scale if dataparser_scale else None,
                        "pipeline_idx": pipeline_idx
                    }
                    for idx, dist in candidates
                ]

                return jsonify(results)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.telemetry_app.route("/telemetry_stream", methods=["GET"])
        def get_telemetry_stream():
            import json
            import time

            def generate():
                while True:
                    try:
                        clients_data = []
                        clients = self.viser_server.get_clients()
                        for client_id, client in clients.items():
                            clients_data.append({
                                "client_id": client_id,
                                "position": client.camera.position.tolist(),
                                "wxyz": client.camera.wxyz.tolist(),
                                "fov": float(client.camera.fov),
                                "aspect": float(client.camera.aspect),
                            })
                        yield f"data: {json.dumps({'clients': clients_data})}\n\n"
                    except GeneratorExit:
                        break
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    time.sleep(0.5)

            response = Response(stream_with_context(generate()), mimetype="text/event-stream")
            response.headers["Cache-Control"] = "no-cache"
            response.headers["X-Accel-Buffering"] = "no"
            return response

        @self.telemetry_app.route("/sync_waypoints", methods=["POST"])
        def sync_waypoints():
            from flask import request as flask_request

            data = flask_request.get_json()
            if data is None:
                return jsonify({"error": "No JSON body"}), 400

            waypoints = data.get("waypoints", [])

            # Remove all existing waypoint scene nodes
            for handles in self.waypoint_handles.values():
                for h in handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
            self.waypoint_handles = {}

            for wp in waypoints:
                name = wp.get("name", "waypoint")
                position = wp.get("position", [0.0, 0.0, 0.0])
                color_hex = wp.get("color", "#ffffff").lstrip("#")
                visible = wp.get("visible", True)

                r = int(color_hex[0:2], 16)
                g = int(color_hex[2:4], 16)
                b = int(color_hex[4:6], 16)

                safe_name = name.replace(" ", "_").replace("/", "_")

                radius = float(wp.get("radius", 0.15))
                sphere_handle = self.viser_server.scene.add_icosphere(
                    name=f"/waypoints/{safe_name}/sphere",
                    radius=radius,
                    color=(r, g, b),
                    position=tuple(float(v) for v in position),
                )
                sphere_handle.visible = visible

                label_handle = self.viser_server.scene.add_label(
                    name=f"/waypoints/{safe_name}/label",
                    text=name,
                    position=tuple(float(v) for v in position),
                )
                label_handle.visible = visible

                self.waypoint_handles[safe_name] = [sphere_handle, label_handle]

            return jsonify({"success": True, "count": len(waypoints)})

        @self.telemetry_app.route("/measure_distance", methods=["POST"])
        def measure_distance():
            from flask import request as flask_request

            data = flask_request.get_json()
            if data is None:
                return jsonify({"error": "No JSON body"}), 400

            # Clear existing measurement visualization
            for h in self.measurement_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self.measurement_handles = []

            pos1 = data.get("pos1")
            pos2 = data.get("pos2")

            if pos1 is None or pos2 is None:
                return jsonify({"success": True, "cleared": True})

            p1 = np.array(pos1, dtype=float)
            p2 = np.array(pos2, dtype=float)

            # Draw yellow spline between the two points
            spline_handle = self.viser_server.scene.add_spline_catmull_rom(
                name="/measurement/line",
                positions=np.array([p1.tolist(), p2.tolist()]),
                color=(255, 220, 0),
                line_width=3.0,
            )
            self.measurement_handles.append(spline_handle)

            # Compute distance
            dist_viser = float(np.linalg.norm(p1 - p2))
            dist_ns = dist_viser / VISER_NERFSTUDIO_SCALE_RATIO

            try:
                dataparser_outputs = self.pipeline.datamanager.train_dataset._dataparser_outputs
                dataparser_scale = float(dataparser_outputs.dataparser_scale)
                dist_meters = dist_ns / dataparser_scale if dataparser_scale else None
            except Exception:
                dist_meters = None

            label_text = f"{dist_meters:.2f} m" if dist_meters is not None else f"{dist_ns:.3f} units"

            midpoint = ((p1 + p2) / 2).tolist()
            label_handle = self.viser_server.scene.add_label(
                name="/measurement/label",
                text=label_text,
                position=tuple(float(v) for v in midpoint),
            )
            self.measurement_handles.append(label_handle)

            return jsonify({
                "success": True,
                "distance_meters": dist_meters,
                "distance_ns": dist_ns,
                "label_text": label_text,
            })

        @self.telemetry_app.route("/probe_mesh", methods=["POST"])
        def probe_mesh():
            try:
                vertices = np.array([
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                    [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]
                ], dtype=np.float32)
                faces = np.array([
                    [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
                ], dtype=np.uint32)
                handle = self.viser_server.scene.add_mesh_simple(
                    name="/probe/mesh",
                    vertices=vertices,
                    faces=faces,
                    color=(100, 200, 100),
                    opacity=0.4,
                    wireframe=False,
                )
                return jsonify({
                    "success": True,
                    "handle_type": type(handle).__name__,
                })
            except Exception as e:
                return jsonify({"error": str(e), "error_type": type(e).__name__}), 500

        @self.telemetry_app.route("/submit_render_job", methods=["POST"])
        def submit_render_job():
            from flask import request as flask_request
            if not self.harvest_api_url or not self.harvest_region_model_id:
                return jsonify({"error": "Viewer not launched via Harvest (HARVEST_API_URL not set)"}), 503

            data = flask_request.get_json()
            if not data:
                return jsonify({"error": "No JSON body"}), 400

            active_region_model_id = (
                self.harvest_region_model_id_b
                if self.active_pipeline_idx == 1 and self.harvest_region_model_id_b
                else self.harvest_region_model_id
            )

            payload = {
                "region_model_id": active_region_model_id,
                "render_name": data.get("render_name"),
                "camera_path_json_path": data.get("camera_path_json_path"),
                "fps": data.get("fps", 30),
                "width": data.get("width", 1920),
                "height": data.get("height", 1080),
                "render_nearest_camera": data.get("render_nearest_camera", False),
            }

            try:
                resp = requests.post(
                    f"{self.harvest_api_url.rstrip('/')}/api/render_jobs",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.harvest_api_token}"},
                    timeout=15,
                )
                return jsonify(resp.json()), resp.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.telemetry_app.route("/submit_export_job", methods=["POST"])
        def submit_export_job():
            from flask import request as flask_request
            if not self.harvest_api_url or not self.harvest_region_model_id:
                return jsonify({"error": "Viewer not launched via Harvest (HARVEST_API_URL not set)"}), 503

            data = flask_request.get_json()
            if not data:
                return jsonify({"error": "No JSON body"}), 400

            active_region_model_id = (
                self.harvest_region_model_id_b
                if self.active_pipeline_idx == 1 and self.harvest_region_model_id_b
                else self.harvest_region_model_id
            )

            payload = {
                "region_model_id": active_region_model_id,
                "export_type": data.get("export_type"),
                "export_name": data.get("export_name"),
                "export_params": data.get("export_params", {}),
            }

            try:
                resp = requests.post(
                    f"{self.harvest_api_url.rstrip('/')}/api/export_jobs",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.harvest_api_token}"},
                    timeout=15,
                )
                return jsonify(resp.json()), resp.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # Start telemetry server in background thread
        def run_telemetry_server():
            self.telemetry_app.run(
                host=config.websocket_host,
                port=self.telemetry_port,
                debug=False,
                use_reloader=False,
                threaded=True
            )

        self.telemetry_thread = threading.Thread(target=run_telemetry_server, daemon=True)
        self.telemetry_thread.start()

        print(f"Telemetry server running at: http://{config.websocket_host}:{self.telemetry_port}/telemetry")

        buttons = (
            viser.theme.TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://nerf.studio",
            ),
            viser.theme.TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/nerfstudio-project/nerfstudio",
            ),
            viser.theme.TitlebarButton(
                text="Documentation",
                icon="Description",
                href="https://docs.nerf.studio",
            ),
        )
        image = viser.theme.TitlebarImage(
            image_url_light="https://docs.nerf.studio/_static/imgs/logo.png",
            image_url_dark="https://docs.nerf.studio/_static/imgs/logo-dark.png",
            image_alt="NerfStudio Logo",
            href="https://docs.nerf.studio/",
        )
        titlebar_theme = viser.theme.TitlebarConfig(buttons=buttons, image=image)
        self.viser_server.gui.configure_theme(
            titlebar_content=titlebar_theme,
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )

        self.render_statemachines: Dict[int, RenderStateMachine] = {}
        self.viser_server.on_client_disconnect(self.handle_disconnect)
        self.viser_server.on_client_connect(self.handle_new_client)

        # Populate the header, which includes the pause button, train cam button, and stats
        self.pause_train = self.viser_server.gui.add_button(
            label="Pause Training", disabled=False, icon=viser.Icon.PLAYER_PAUSE_FILLED
        )
        self.pause_train.on_click(lambda _: self.toggle_pause_button())
        self.pause_train.on_click(lambda han: self._toggle_training_state(han))
        self.resume_train = self.viser_server.gui.add_button(
            label="Resume Training", disabled=False, icon=viser.Icon.PLAYER_PLAY_FILLED
        )
        self.resume_train.on_click(lambda _: self.toggle_pause_button())
        self.resume_train.on_click(lambda han: self._toggle_training_state(han))
        if self.train_btn_state == "training":
            self.resume_train.visible = False
        else:
            self.pause_train.visible = False

        # Add buttons to toggle training image visibility
        self.hide_images = self.viser_server.gui.add_button(
            label="Hide Train Cams", disabled=False, icon=viser.Icon.EYE_OFF, color=None
        )
        self.hide_images.on_click(lambda _: self.set_camera_visibility(False))
        self.hide_images.on_click(lambda _: self.toggle_cameravis_button())
        self.show_images = self.viser_server.gui.add_button(
            label="Show Train Cams", disabled=False, icon=viser.Icon.EYE, color=None
        )
        self.show_images.on_click(lambda _: self.set_camera_visibility(True))
        self.show_images.on_click(lambda _: self.toggle_cameravis_button())
        self.show_images.visible = False
        mkdown = self.make_stats_markdown(0, "0x0px")
        self.stats_markdown = self.viser_server.gui.add_markdown(mkdown)
        tabs = self.viser_server.gui.add_tab_group()
        control_tab = tabs.add_tab("Control", viser.Icon.SETTINGS)
        with control_tab:
            self.control_panel = ControlPanel(
                self.viser_server,
                self.include_time,
                VISER_NERFSTUDIO_SCALE_RATIO,
                self._trigger_rerender,
                self._output_type_change,
                self._output_split_type_change,
                default_composite_depth=self.config.default_composite_depth,
            )
        # Render and Export tabs disabled - only show Control tab
        config_path = self.log_filename.parents[0] / "config.yml"
        with tabs.add_tab("Render", viser.Icon.CAMERA):
            self.render_tab_state=populate_render_tab(self.viser_server, config_path, self.datapath, self.control_panel, harvest_telem_port=self.telemetry_port)
        with tabs.add_tab("Export", viser.Icon.PACKAGE_EXPORT):
            populate_export_tab(self.viser_server, self.control_panel, config_path, self.pipeline.model, harvest_telem_port=self.telemetry_port)

        #from nerfstudio.viewer.render_panel import RenderTabState
        #self.render_tab_state = RenderTabState(
        #    preview_render=False,
        #    preview_fov=75.0,
        #    preview_time=0.0,
        #    preview_aspect=1.0,
        #    preview_camera_type="Perspective"
        #)

        # Keep track of the pointers to generated GUI folders, because each generated folder holds a unique ID.
        viewer_gui_folders = dict()

        def prev_cb_wrapper(prev_cb):
            # We wrap the callbacks in the train_lock so that the callbacks are thread-safe with the
            # concurrently executing render thread. This may block rendering, however this can be necessary
            # if the callback uses get_outputs internally.
            def cb_lock(element):
                with self.train_lock if self.train_lock is not None else contextlib.nullcontext():
                    prev_cb(element)

            return cb_lock

        def nested_folder_install(folder_labels: List[str], prev_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
                # also rewire the hook to rerender
                prev_cb = element.cb_hook
                element.cb_hook = lambda element: [prev_cb_wrapper(prev_cb)(element), self._trigger_rerender()]
            else:
                # recursively create folders
                # If the folder name is "Custom Elements/a/b", then:
                #   in the beginning: folder_path will be
                #       "/".join([] + ["Custom Elements"]) --> "Custom Elements"
                #   later, folder_path will be
                #       "/".join(["Custom Elements"] + ["a"]) --> "Custom Elements/a"
                #       "/".join(["Custom Elements", "a"] + ["b"]) --> "Custom Elements/a/b"
                #  --> the element will be installed in the folder "Custom Elements/a/b"
                #
                # Note that the gui_folder is created only when the folder is not in viewer_gui_folders,
                # and we use the folder_path as the key to check if the folder is already created.
                # Otherwise, use the existing folder as context manager.
                folder_path = "/".join(prev_labels + [folder_labels[0]])
                if folder_path not in viewer_gui_folders:
                    viewer_gui_folders[folder_path] = self.viser_server.gui.add_folder(folder_labels[0])
                with viewer_gui_folders[folder_path]:
                    nested_folder_install(folder_labels[1:], prev_labels + [folder_labels[0]], element)

        with control_tab:
            from nerfstudio.viewer_legacy.server.viewer_elements import ViewerElement as LegacyViewerElement

            if len(parse_object(pipeline, LegacyViewerElement, "Custom Elements")) > 0:
                from nerfstudio.utils.rich_utils import CONSOLE

                CONSOLE.print(
                    "Legacy ViewerElements detected in model, please import nerfstudio.viewer.viewer_elements instead",
                    style="bold yellow",
                )
            self.viewer_elements = []
            self.viewer_elements.extend(parse_object(pipeline, ViewerElement, "Custom Elements"))
            for param_path, element in self.viewer_elements:
                folder_labels = param_path.split("/")[:-1]
                nested_folder_install(folder_labels, [], element)

            # scrape the trainer/pipeline for any ViewerControl objects to initialize them
            self.viewer_controls: List[ViewerControl] = [
                e for (_, e) in parse_object(pipeline, ViewerControl, "Custom Elements")
            ]

            # Add comparison mode toggle button
            if self.pipeline_b is not None:
                self.compare_toggle = self.viser_server.gui.add_button(
                    label="Swap Models",
                    disabled=False,
                    icon=viser.Icon.ARROWS_LEFT_RIGHT,
                )
                self.compare_toggle.on_click(lambda _: self._toggle_comparison_model())
                self.compare_status = self.viser_server.gui.add_markdown(
                    f"**Active Model:** {self.experiment_name_a}"
                )

                # Add model alignment controls
                with self.viser_server.gui.add_folder("Model B Alignment"):
                    self.compare_trans = ViewerVec3(
                        "Translation",
                        (0.0, 0.0, 0.0),
                        step=0.1,
                        cb_hook=lambda _: self._trigger_rerender(),
                        hint="Translate model B relative to model A (x, y, z in meters)",
                    )
                    self.compare_rot = ViewerVec3(
                        "Rotation",
                        (0.0, 0.0, 0.0),
                        step=0.01,
                        cb_hook=lambda _: self._trigger_rerender(),
                        hint="Rotate model B in radians (roll, pitch, yaw)",
                    )
                    self.compare_reset = self.viser_server.gui.add_button(
                        label="Reset Transform",
                        disabled=False,
                        icon=viser.Icon.ARROW_BACK_UP,
                    )
                    self.compare_reset.on_click(lambda _: self._reset_comparison_transform())

                    # Install elements
                    self.compare_trans.install(self.viser_server)
                    self.compare_rot.install(self.viser_server)

        for c in self.viewer_controls:
            c._setup(self)

        # Diagnostics for Gaussian Splatting: where the points are at the start of training.
        # This is hidden by default, it can be shown from the Viser UI's scene tree table.
        # Skip in comparison mode to avoid showing stale data after model swap.
        if isinstance(pipeline.model, SplatfactoModel) and self.pipeline_b is None:
            self.viser_server.scene.add_point_cloud(
                "/gaussian_splatting_initial_points",
                points=pipeline.model.means.numpy(force=True) * VISER_NERFSTUDIO_SCALE_RATIO,
                colors=(255, 0, 0),
                point_size=0.01,
                point_shape="circle",
                visible=False,  # Hidden by default.
            )
        self.ready = True

    def toggle_pause_button(self) -> None:
        self.pause_train.visible = not self.pause_train.visible
        self.resume_train.visible = not self.resume_train.visible

    def toggle_cameravis_button(self) -> None:
        self.hide_images.visible = not self.hide_images.visible
        self.show_images.visible = not self.show_images.visible

    def _toggle_comparison_model(self) -> None:
        """Toggle between model A and model B."""
        if self.pipeline_b is None:
            return

        # Swap active index
        self.active_pipeline_idx = 1 - self.active_pipeline_idx

        # Update status to show current active model
        if self.active_pipeline_idx == 0:
            self.compare_status.content = f"**Active Model:** {self.experiment_name_a}"
            # Show model A cameras, hide model B cameras
            for handle in self.camera_handles_a.values():
                handle.visible = True
            for handle in self.camera_handles_b.values():
                handle.visible = False
        else:
            self.compare_status.content = f"**Active Model:** {self.experiment_name_b}"
            # Show model B cameras, hide model A cameras
            for handle in self.camera_handles_a.values():
                handle.visible = False
            for handle in self.camera_handles_b.values():
                handle.visible = True

        # Trigger rerender for all clients
        self._trigger_rerender()

    def _reset_comparison_transform(self) -> None:
        """Reset model B alignment transform to zero."""
        if self.compare_trans is not None:
            self.compare_trans.value = (0.0, 0.0, 0.0)
        if self.compare_rot is not None:
            self.compare_rot.value = (0.0, 0.0, 0.0)
        self._trigger_rerender()

    def make_stats_markdown(self, step: Optional[int], res: Optional[str]) -> str:
        # if either are None, read it from the current stats_markdown content
        if step is None:
            step = int(self.stats_markdown.content.split("\n")[0].split(": ")[1])
        if res is None:
            res = (self.stats_markdown.content.split("\n")[1].split(": ")[1]).strip()
        return f"Step: {step}  \nResolution: {res}"

    def update_step(self, step):
        """
        Args:
            step: the train step to set the model to
        """
        self.stats_markdown.content = self.make_stats_markdown(step, None)

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        R = vtf.SO3(wxyz=client.camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(client.camera.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
        c2w = torch.concatenate([R, pos[:, None]], dim=1)
        if self.ready and self.render_tab_state.preview_render:
            camera_type = self.render_tab_state.preview_camera_type
            camera_state = CameraState(
                fov=self.render_tab_state.preview_fov,
                aspect=self.render_tab_state.preview_aspect,
                c2w=c2w,
                time=self.render_tab_state.preview_time,
                camera_type=CameraType.PERSPECTIVE
                if camera_type == "Perspective"
                else CameraType.FISHEYE
                if camera_type == "Fisheye"
                else CameraType.EQUIRECTANGULAR
                if camera_type == "Equirectangular"
                else assert_never(camera_type),
                idx=self.current_camera_idx,
            )
        else:
            camera_state = CameraState(
                fov=client.camera.fov,
                aspect=client.camera.aspect,
                c2w=c2w,
                camera_type=CameraType.PERSPECTIVE,
                idx=self.current_camera_idx,
            )
        return camera_state

    def handle_disconnect(self, client: viser.ClientHandle) -> None:
        self.render_statemachines[client.client_id].running = False
        self.render_statemachines.pop(client.client_id)

    def handle_new_client(self, client: viser.ClientHandle) -> None:
        self.render_statemachines[client.client_id] = RenderStateMachine(self, VISER_NERFSTUDIO_SCALE_RATIO, client)
        self.render_statemachines[client.client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            if not self.ready:
                return
            self.last_move_time = time.time()
            with self.viser_server.atomic():
                camera_state = self.get_camera_state(client)
                self.render_statemachines[client.client_id].action(RenderAction("move", camera_state))

    def set_camera_visibility(self, visible: bool) -> None:
        """Toggle the visibility of the training cameras for the active model."""
        with self.viser_server.atomic():
            # Only toggle cameras for the active model
            camera_handles = self.camera_handles_a if self.active_pipeline_idx == 0 else self.camera_handles_b
            for idx in camera_handles:
                camera_handles[idx].visible = visible

    def update_camera_poses(self):
        # TODO this fn accounts for like ~5% of total train time
        # Update the train camera locations based on optimization
        active_model = self.get_model()
        if hasattr(active_model, "camera_optimizer"):
            camera_optimizer = active_model.camera_optimizer
        else:
            return

        # Get active camera handles and original c2w based on which model is active
        if self.active_pipeline_idx == 0:
            camera_handles = self.camera_handles_a
            original_c2w = self.original_c2w_a
        else:
            camera_handles = self.camera_handles_b
            original_c2w = self.original_c2w_b

        idxs = list(camera_handles.keys())
        with torch.no_grad():
            assert isinstance(camera_optimizer, CameraOptimizer)
            c2ws_delta = camera_optimizer(torch.tensor(idxs, device=camera_optimizer.device)).cpu().numpy()
        for i, key in enumerate(idxs):
            # both are numpy arrays
            c2w_orig = original_c2w[key]
            c2w_delta = c2ws_delta[i, ...]
            c2w = c2w_orig @ np.concatenate((c2w_delta, np.array([[0, 0, 0, 1]])), axis=0)
            R = vtf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
            R = R @ vtf.SO3.from_x_radians(np.pi)
            camera_handles[key].position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
            camera_handles[key].wxyz = R.wxyz

    def _trigger_rerender(self) -> None:
        """Interrupt current render."""
        if not self.ready:
            return
        clients = self.viser_server.get_clients()
        for id in clients:
            camera_state = self.get_camera_state(clients[id])
            self.render_statemachines[id].action(RenderAction("move", camera_state))

    def _toggle_training_state(self, _) -> None:
        """Toggle the trainer's training state."""
        if self.trainer is not None:
            if self.trainer.training_state == "training":
                self.trainer.training_state = "paused"
            elif self.trainer.training_state == "paused":
                self.trainer.training_state = "training"

    def _output_type_change(self, _):
        self.output_type_changed = True

    def _output_split_type_change(self, _):
        self.output_split_type_changed = True

    def _pick_drawn_image_idxs(self, total_num: int) -> list[int]:
        """Determine indicies of images to display in viewer.

        Args:
            total_num: total number of training images.

        Returns:
            List of indices from [0, total_num-1].
        """
        if self.config.max_num_display_images < 0:
            num_display_images = total_num
        else:
            num_display_images = min(self.config.max_num_display_images, total_num)
        # draw indices, roughly evenly spaced
        return np.linspace(0, total_num - 1, num_display_images, dtype=np.int32).tolist()

    def _create_camera_frustums(
        self, train_dataset: InputDataset, camera_prefix: str, visible: bool = True
    ) -> tuple[Dict[int, viser.CameraFrustumHandle], Dict[int, np.ndarray]]:
        """Create camera frustums for a dataset.

        Args:
            train_dataset: dataset containing cameras to render
            camera_prefix: prefix for camera names (e.g., "/cameras_a/" or "/cameras_b/")
            visible: whether cameras should be visible initially

        Returns:
            Tuple of (camera_handles dict, original_c2w dict)
        """
        camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        original_c2w: Dict[int, np.ndarray] = {}
        image_indices = self._pick_drawn_image_idxs(len(train_dataset))
        for idx in image_indices:
            image = train_dataset[idx]["image"]
            camera = train_dataset.cameras[idx]
            image_uint8 = (image * 255).detach().type(torch.uint8)
            image_uint8 = image_uint8.permute(2, 0, 1)

            # torchvision can be slow to import, so we do it lazily.
            import torchvision

            image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias=None)  # type: ignore
            image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            c2w = camera.camera_to_worlds.cpu().numpy()
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            camera_handle = self.viser_server.scene.add_camera_frustum(
                name=f"{camera_prefix}camera_{idx:05d}",
                fov=float(2 * np.arctan((camera.cx / camera.fx[0]).cpu())),
                scale=self.config.camera_frustum_scale,
                aspect=float((camera.cx[0] / camera.cy[0]).cpu()),
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
                visible=visible,
            )

            def create_on_click_callback(capture_idx):
                def on_click_callback(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                    with event.client.atomic():
                        event.client.camera.position = event.target.position
                        event.client.camera.wxyz = event.target.wxyz
                        self.current_camera_idx = capture_idx

                return on_click_callback

            camera_handle.on_click(create_on_click_callback(idx))

            camera_handles[idx] = camera_handle
            original_c2w[idx] = c2w

        return camera_handles, original_c2w

    def init_scene(
        self,
        train_dataset: InputDataset,
        train_state: Literal["training", "paused", "completed"],
        eval_dataset: Optional[InputDataset] = None,
    ) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            train_state: Current status of training
        """
        # Create camera frustums for model A
        self.camera_handles_a, self.original_c2w_a = self._create_camera_frustums(
            train_dataset, "/cameras_a/", visible=True
        )

        # Create camera frustums for model B if in comparison mode
        if self.pipeline_b is not None:
            self.camera_handles_b, self.original_c2w_b = self._create_camera_frustums(
                self.pipeline_b.datamanager.train_dataset, "/cameras_b/", visible=False
            )
        else:
            self.camera_handles_b = {}
            self.original_c2w_b = {}

        self.train_state = train_state
        self.train_util = 0.9

    def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            num_rays_per_batch: number of rays per batch, used during training
        """
        self.step = step

        if len(self.render_statemachines) == 0:
            return
        # this stops training while moving to make the response smoother
        while time.time() - self.last_move_time < 0.1:
            time.sleep(0.05)
        if self.trainer is not None and self.trainer.training_state == "training" and self.train_util != 1:
            if (
                EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
                and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
            ):
                train_s = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                vis_s = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
                train_util = self.train_util
                vis_n = self.control_panel.max_res**2
                train_n = num_rays_per_batch
                train_time = train_n / train_s
                vis_time = vis_n / vis_s

                render_freq = train_util * vis_time / (train_time - train_util * train_time)
            else:
                render_freq = 30
            if step > self.last_step + render_freq:
                self.last_step = step
                clients = self.viser_server.get_clients()
                for id in clients:
                    camera_state = self.get_camera_state(clients[id])
                    if camera_state is not None:
                        self.render_statemachines[id].action(RenderAction("step", camera_state))
                self.update_camera_poses()
                self.update_step(step)

    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_type_changed:
            self.control_panel.update_colormap_options(dimensions, dtype)
            self.output_type_changed = False

    def update_split_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_split_type_changed:
            self.control_panel.update_split_colormap_options(dimensions, dtype)
            self.output_split_type_changed = False

    def get_model(self) -> Model:
        """Returns the active model."""
        if self.active_pipeline_idx == 1 and self.pipeline_b is not None:
            return self.pipeline_b.model
        return self.pipeline.model

    def apply_model_b_transform(self, camera: Cameras) -> Cameras:
        """Apply transform to camera for model_b coordinate alignment.

        Transforms camera from viewer space to model_b's coordinate system.

        Args:
            camera: Camera in viewer coordinate system

        Returns:
            Camera transformed to model_b coordinate system
        """
        if self.compare_trans is None or self.compare_rot is None:
            return camera

        # Build 4x4 transform from UI values
        trans = self.compare_trans.value
        rot = self.compare_rot.value

        R = torch.tensor(
            vtf.SO3.from_rpy_radians(rot[0], rot[1], rot[2]).as_matrix(),
            dtype=torch.float32,
            device=camera.device,
        )
        T = torch.tensor(trans, dtype=torch.float32, device=camera.device)

        H = torch.eye(4, dtype=torch.float32, device=camera.device)
        H[:3, :3] = R
        H[:3, 3] = T

        # Transform camera_to_worlds: c2w_new = H @ c2w_old
        c2w = camera.camera_to_worlds[0]  # [3, 4]
        c2w_homogeneous = torch.cat(
            [c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=camera.device)], dim=0
        )  # [4, 4]

        c2w_transformed = (H @ c2w_homogeneous)[:3, :]  # [3, 4]
        camera.camera_to_worlds[0] = c2w_transformed

        return camera

    def training_complete(self) -> None:
        """Called when training is complete."""
        self.training_state = "completed"
