"""Browser client for the simulator-independent visualization server."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
import json

from .buttons import VisualizationButton, normalize_buttons
from .dashboard import VisualizationCurve, normalize_curves
from .presentation import (
    BrowserSceneStyle,
    VisualizationTrail,
    normalize_trails,
)
from .three_scene import BrowserThreeScene
from .theme import CAMERA_DRAG_MODES, CAMERA_HELP


_THREE_RENDERER_MODULE = r"""
  <script type="module">
    const status = document.createElement('div');
    status.className = 'three-renderer-status';
    status.textContent = '正在加载彩色 Three.js 场景…';
    document.getElementById('viewport').appendChild(status);

    function failThreeRenderer(error) {
      console.error('Three.js scene initialization failed', error);
      status.dataset.state = 'error';
      status.textContent = `彩色场景加载失败：${error instanceof Error ? error.message : error}`;
    }

    async function startThreeRenderer() {
      const [THREE, gltfModule, objModule, stlModule, urdfModule] =
        await Promise.all([
          import('https://esm.sh/three@0.160.1'),
          import('https://esm.sh/three@0.160.1/examples/jsm/loaders/GLTFLoader.js'),
          import('https://esm.sh/three@0.160.1/examples/jsm/loaders/OBJLoader.js'),
          import('https://esm.sh/three@0.160.1/examples/jsm/loaders/STLLoader.js'),
          import('https://esm.sh/urdf-loader@0.12.4?deps=three@0.160.1')
        ]);
      const {GLTFLoader} = gltfModule;
      const {OBJLoader} = objModule;
      const {STLLoader} = stlModule;
      const URDFLoader = urdfModule.default ?? urdfModule.URDFLoader ?? urdfModule;
      const config = window.visualizationThreeSceneConfig;
      const style = window.visualizationSceneStyle;
      const trailDefinitions = window.visualizationTrajectoryDefinitions;
      if (config === null || typeof config !== 'object') {
        throw new Error('missing browser Three.js scene contract');
      }

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(style?.background_color ?? '#0B0F15');

      const camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 100);
      camera.up.set(0, 0, 1);
      const renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
        preserveDrawingBuffer: false
      });
      renderer.setPixelRatio(1);

      const hemi = new THREE.HemisphereLight(
        '#F7FBFF',
        '#5B6F86',
        2.8
      );
      scene.add(hemi);

      function addDirectional(color, intensity, position, castsShadow = false) {
        const light = new THREE.DirectionalLight(color, intensity);
        light.position.set(...position);
        light.castShadow = castsShadow;
        if (castsShadow) {
          light.shadow.mapSize.set(2048, 2048);
          light.shadow.camera.near = 0.1;
          light.shadow.camera.far = 8;
          light.shadow.camera.left = -2.2;
          light.shadow.camera.right = 2.2;
          light.shadow.camera.top = 2.2;
          light.shadow.camera.bottom = -2.2;
          light.shadow.bias = -0.00012;
        }
        scene.add(light);
      }
      addDirectional(style?.key_light_color ?? '#FFFFFF', 2.1, [2.5, -2.0, 3.8]);
      addDirectional(style?.fill_light_color ?? '#F6FAFF', 1.4, [-3.5, -1.0, 2.8]);
      addDirectional(style?.rim_light_color ?? '#E8F3FF', 1.0, [0.5, 3.0, 2.5]);
      const topLight = new THREE.PointLight('#FFFFFF', 1.2, 0, 2);
      topLight.position.set(0.0, 0.6, 2.2);
      scene.add(topLight);

      const floorZ = Number(style?.grid_plane_z_m ?? 0);
      const gridSize = Number(style?.grid_size_m ?? 6);
      const gridDivisions = Number(style?.grid_divisions ?? 24);
      const floor = new THREE.Mesh(
        new THREE.CircleGeometry(3.5, 96),
        new THREE.MeshStandardMaterial({
          color: style?.floor_tint ?? '#2D3B4E',
          roughness: 0.98,
          metalness: 0.04
        })
      );
      floor.position.z = floorZ - 0.001;
      floor.receiveShadow = true;
      scene.add(floor);
      const grid = new THREE.GridHelper(
        gridSize,
        gridDivisions,
        style?.grid_major_color ?? '#35557A',
        style?.grid_minor_color ?? '#203447'
      );
      grid.rotation.x = Math.PI / 2;
      grid.position.z = floorZ;
      scene.add(grid);
      const axes = new THREE.AxesHelper(0.35);
      scene.add(axes);

      const actors = new Map();

      function materialFor(definition, color) {
        const opacity = definition.opacity ?? 1.0;
        const materialColor = Array.isArray(color)
          ? new THREE.Color(...color)
          : color;
        return new THREE.MeshStandardMaterial({
          color: materialColor,
          roughness: definition.roughness,
          metalness: definition.metalness,
          transparent: opacity < 1.0,
          opacity,
          depthWrite: opacity >= 1.0,
          wireframe: Boolean(definition.wireframe)
        });
      }

      for (const definition of config.primitive_actors) {
        let geometry;
        if (definition.geometry === 'box') {
          geometry = new THREE.BoxGeometry(...definition.dimensions_m);
        } else if (definition.geometry === 'cylinder') {
          geometry = new THREE.CylinderGeometry(
            definition.dimensions_m[0],
            definition.dimensions_m[0],
            definition.dimensions_m[1],
            96
          );
          geometry.rotateX(Math.PI / 2);
        } else {
          throw new Error(`unsupported primitive geometry: ${definition.geometry}`);
        }
        const mesh = new THREE.Mesh(
          geometry,
          materialFor(definition, definition.color)
        );
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.visible = false;
        scene.add(mesh);
        actors.set(definition.actor_id, {
          root: mesh,
          definition,
          robot: null,
          ready: true,
          hasState: false,
          stateVisible: true
        });
      }

      function meshFormat(path) {
        return new URL(path, window.location.href).pathname.split('.').pop().toLowerCase();
      }

      function installColorMaterials(robot, definition) {
        if (!robot.links) return;
        const hasOverrides = Object.keys(definition.link_colors).length > 0;
        const linkByObject = new Map(
          Object.entries(robot.links).map(([name, object]) => [object, name])
        );
        const materials = new Map();
        const colorFor = linkName =>
          definition.link_colors[linkName] ?? definition.fallback_color;
        robot.traverse(child => {
          if (!child.isMesh) return;
          child.castShadow = true;
          child.receiveShadow = true;
          if (!hasOverrides) return;
          let ancestor = child.parent;
          while (ancestor !== null && !linkByObject.has(ancestor)) {
            ancestor = ancestor.parent;
          }
          const linkName = ancestor === null ? '' : linkByObject.get(ancestor);
          const color = colorFor(linkName);
          let material = materials.get(color);
          if (material === undefined) {
            material = materialFor(definition, color);
            materials.set(color, material);
          }
          child.material = material;
        });
      }

      let pendingUrdfActors = config.urdf_actors.length;
      let sceneAssetFailure = false;
      let readyStatusScheduled = false;

      function markSceneAssetsReady() {
        window.visualizationThreeAssetsReady = true;
        window.dispatchEvent(new Event('visualization-three-assets-ready'));
        if (!sceneAssetFailure) {
          status.textContent = '模型资产已加载，正在同步场景状态…';
        }
      }

      function loadUrdfActor(definition) {
        const manager = new THREE.LoadingManager();
        const loader = new URDFLoader(manager);
        let robot = null;
        let handle = null;
        let finished = false;
        const finish = () => {
          if (finished || robot === null || handle === null) return;
          finished = true;
          installColorMaterials(robot, definition);
          handle.ready = true;
          robot.visible = handle.hasState && handle.stateVisible;
          pendingUrdfActors -= 1;
          if (pendingUrdfActors === 0) markSceneAssetsReady();
          window.visualizationRequestRedraw?.();
        };
        manager.onProgress = (_url, loaded, total) => {
          status.textContent = `正在加载机器人资产：${loaded}/${total}`;
        };
        manager.onLoad = finish;
        manager.onError = url => {
          sceneAssetFailure = true;
          status.dataset.state = 'error';
          status.textContent = `机器人可视资产加载失败：${url}`;
        };
        loader.packages = '';
        loader.loadMeshCb = (path, loadingManager, onComplete) => {
          const format = meshFormat(path);
          const complete = (object, error = null) => {
            onComplete(object, error);
          };
          if (format === 'glb' || format === 'gltf') {
            new GLTFLoader(loadingManager).load(
              path,
              result => complete(result.scene),
              undefined,
              error => complete(null, error)
            );
          } else if (format === 'stl') {
            new STLLoader(loadingManager).load(
              path,
              geometry => {
                if (geometry.getAttribute('normal') === undefined) {
                  geometry.computeVertexNormals();
                }
                complete(new THREE.Mesh(geometry, materialFor(definition, definition.fallback_color)));
              },
              undefined,
              error => complete(null, error)
            );
          } else if (format === 'obj') {
            new OBJLoader(loadingManager).load(
              path,
              object => complete(object),
              undefined,
              error => complete(null, error)
            );
          } else {
            complete(null, new Error(`unsupported robot mesh format: ${format}`));
          }
        };

        const documentNode = new DOMParser().parseFromString(
          definition.urdf_xml,
          'application/xml'
        );
        const parserError = documentNode.querySelector('parsererror');
        if (parserError !== null) throw new Error(parserError.textContent);
        robot = loader.parse(documentNode);
        robot.visible = false;
        robot.name = definition.actor_id;
        scene.add(robot);
        handle = {
          root: robot,
          definition,
          robot,
          ready: false,
          hasState: false,
          stateVisible: true
        };
        actors.set(definition.actor_id, handle);
        if (documentNode.querySelectorAll('visual mesh').length === 0) {
          queueMicrotask(finish);
        }
      }

      for (const definition of config.urdf_actors) loadUrdfActor(definition);
      if (pendingUrdfActors === 0) markSceneAssetsReady();

      const trailHandles = new Map();
      for (const definition of trailDefinitions) {
        const material = new THREE.MeshBasicMaterial({
          color: definition.color,
          depthTest: false,
          depthWrite: false
        });
        const group = new THREE.Group();
        const markers = [];
        const markerGeometry = new THREE.SphereGeometry(
          definition.marker_radius_m * 0.48,
          16,
          12
        );
        for (let index = 0; index < 18; index += 1) {
          const marker = new THREE.Mesh(markerGeometry, material);
          marker.renderOrder = 30;
          marker.visible = false;
          group.add(marker);
          markers.push(marker);
        }
        const endpoint = new THREE.Mesh(
          new THREE.SphereGeometry(definition.marker_radius_m, 24, 16),
          material
        );
        endpoint.visible = false;
        endpoint.renderOrder = 30;
        group.add(endpoint);
        scene.add(group);
        trailHandles.set(definition.trail_id, {
          definition,
          group,
          material,
          markers,
          endpoint,
          tube: null,
          appliedSequence: -1
        });
      }

      const polylineOverlayHandles = new Map();
      let polylineOverlayOrder = 0;
      for (const definition of config.polyline_overlays ?? []) {
        const maximumLineVertices = definition.max_polylines *
          Math.max(0, definition.max_points_per_polyline - 1) * 2;
        const maximumPointVertices = definition.max_polylines *
          definition.max_points_per_polyline;
        const lineGeometry = new THREE.BufferGeometry();
        const linePosition = new THREE.BufferAttribute(
          new Float32Array(maximumLineVertices * 3),
          3
        );
        linePosition.setUsage(THREE.DynamicDrawUsage);
        lineGeometry.setAttribute('position', linePosition);
        lineGeometry.setDrawRange(0, 0);
        const pointGeometry = new THREE.BufferGeometry();
        const pointPosition = new THREE.BufferAttribute(
          new Float32Array(maximumPointVertices * 3),
          3
        );
        pointPosition.setUsage(THREE.DynamicDrawUsage);
        pointGeometry.setAttribute('position', pointPosition);
        pointGeometry.setDrawRange(0, 0);
        const lineMaterial = new THREE.LineBasicMaterial({
          color: definition.color,
          transparent: definition.opacity < 1.0,
          opacity: definition.opacity,
          depthTest: definition.depth_test,
          depthWrite: false
        });
        const pointMaterial = new THREE.PointsMaterial({
          color: definition.color,
          size: definition.point_size_px,
          sizeAttenuation: false,
          transparent: definition.opacity < 1.0,
          opacity: definition.opacity,
          depthTest: definition.depth_test,
          depthWrite: false
        });
        const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
        const points = new THREE.Points(pointGeometry, pointMaterial);
        lines.frustumCulled = false;
        points.frustumCulled = false;
        lines.renderOrder = 40 + polylineOverlayOrder * 2;
        points.renderOrder = 41 + polylineOverlayOrder * 2;
        polylineOverlayOrder += 1;
        const group = new THREE.Group();
        group.add(lines, points);
        scene.add(group);
        polylineOverlayHandles.set(definition.overlay_id, {
          definition,
          group,
          lineGeometry,
          linePosition,
          pointGeometry,
          pointPosition,
          contentVisible: false,
          controlVisible: true,
          appliedSequence: -1
        });
      }

      function syncPolylineOverlayVisibility(handle) {
        handle.group.visible = handle.contentVisible && handle.controlVisible;
      }

      const overlayVisibilityControls = config.overlay_visibility_controls ?? [];
      if (overlayVisibilityControls.length) {
        const controls = document.createElement('div');
        controls.className = 'overlay-visibility-controls';
        controls.setAttribute('role', 'group');
        controls.setAttribute('aria-label', 'Object Flow 显示控制');
        controls.addEventListener('pointerdown', event => event.stopPropagation());
        for (const definition of overlayVisibilityControls) {
          const button = document.createElement('button');
          button.type = 'button';
          button.id = `overlay-visibility-${definition.control_id}`;
          button.dataset.controlId = definition.control_id;
          let visible = definition.default_visible !== false;
          const applyVisibility = () => {
            for (const overlayId of definition.overlay_ids) {
              const handle = polylineOverlayHandles.get(overlayId);
              if (handle === undefined) {
                throw new Error(`unknown controlled polyline overlay: ${overlayId}`);
              }
              handle.controlVisible = visible;
              syncPolylineOverlayVisibility(handle);
            }
            button.dataset.visible = visible ? 'true' : 'false';
            button.setAttribute('aria-pressed', visible ? 'true' : 'false');
            button.textContent = `${visible ? '关闭' : '开启'}${definition.label}`;
            window.visualizationRequestRedraw?.();
          };
          button.addEventListener('click', () => {
            visible = !visible;
            applyVisibility();
          });
          controls.appendChild(button);
          applyVisibility();
        }
        document.getElementById('viewport').appendChild(controls);
      }

      function cssColor(value) {
        if (!Array.isArray(value)) return value;
        return `rgb(${value.map(component => Math.round(component * 255)).join(',')})`;
      }

      const legendDefinitions = [
        ...(config.primitive_actors ?? []).filter(definition => definition.label),
        ...(config.polyline_overlays ?? [])
      ];
      if (legendDefinitions.length) {
        const legend = document.createElement('div');
        legend.className = 'three-scene-legend';
        for (const definition of legendDefinitions) {
          const item = document.createElement('span');
          const swatch = document.createElement('i');
          swatch.style.background = cssColor(definition.color);
          item.appendChild(swatch);
          item.appendChild(document.createTextNode(definition.label));
          legend.appendChild(item);
        }
        document.getElementById('viewport').appendChild(legend);
      }

      function sampledVectors(rawPoints, limit = 240) {
        const stride = Math.max(1, Math.ceil(rawPoints.length / limit));
        const selected = [];
        for (let index = 0; index < rawPoints.length; index += stride) {
          selected.push(new THREE.Vector3(...rawPoints[index]));
        }
        if (rawPoints.length && (rawPoints.length - 1) % stride !== 0) {
          selected.push(new THREE.Vector3(...rawPoints[rawPoints.length - 1]));
        }
        return selected;
      }

      function updateTrail(handle, rawPoints) {
        const points = sampledVectors(Array.isArray(rawPoints) ? rawPoints : []);
        if (handle.tube !== null) {
          handle.group.remove(handle.tube);
          handle.tube.geometry.dispose();
          handle.tube = null;
        }
        if (points.length >= 2) {
          const curve = points.length === 2
            ? new THREE.LineCurve3(points[0], points[1])
            : new THREE.CatmullRomCurve3(points, false, 'centripetal');
          const geometry = new THREE.TubeGeometry(
            curve,
            Math.min(384, Math.max(8, points.length * 3)),
            handle.definition.tube_radius_m,
            8,
            false
          );
          handle.tube = new THREE.Mesh(geometry, handle.material);
          handle.tube.renderOrder = 30;
          handle.group.add(handle.tube);
        }
        for (const marker of handle.markers) marker.visible = false;
        if (points.length) {
          const stride = Math.max(1, Math.floor(points.length / handle.markers.length));
          let markerIndex = 0;
          for (let index = 0; index < points.length - 1 && markerIndex < handle.markers.length; index += stride) {
            const marker = handle.markers[markerIndex++];
            marker.position.copy(points[index]);
            marker.visible = true;
          }
          handle.endpoint.position.copy(points[points.length - 1]);
          handle.endpoint.visible = true;
        } else {
          handle.endpoint.visible = false;
        }
      }

      function updatePolylineOverlay(handle, rawPolylines) {
        const polylines = Array.isArray(rawPolylines) ? rawPolylines : [];
        const lineArray = handle.linePosition.array;
        const pointArray = handle.pointPosition.array;
        let lineVertexCount = 0;
        let pointVertexCount = 0;
        for (const polyline of polylines) {
          if (!Array.isArray(polyline)) continue;
          for (const point of polyline) {
            pointArray.set(point, pointVertexCount * 3);
            pointVertexCount += 1;
          }
          for (let index = 1; index < polyline.length; index += 1) {
            lineArray.set(polyline[index - 1], lineVertexCount * 3);
            lineVertexCount += 1;
            lineArray.set(polyline[index], lineVertexCount * 3);
            lineVertexCount += 1;
          }
        }
        handle.lineGeometry.setDrawRange(0, lineVertexCount);
        handle.pointGeometry.setDrawRange(0, pointVertexCount);
        handle.linePosition.needsUpdate = true;
        handle.pointPosition.needsUpdate = true;
        handle.contentVisible = pointVertexCount > 0;
        syncPolylineOverlayVisibility(handle);
      }

      let appliedRenderRevision = -1;
      let appliedActorSequence = -1;
      function applySnapshot() {
        const snapshot = window.visualizationPresentationSnapshot;
        const renderRevision = snapshot?.render_revision ?? snapshot?.sequence;
        if (snapshot === null || snapshot === undefined ||
            renderRevision === appliedRenderRevision) {
          return;
        }
        appliedRenderRevision = renderRevision;
        if (snapshot.view !== null) {
          camera.position.set(...snapshot.view.eye);
          camera.up.set(0, 0, 1);
          camera.lookAt(...snapshot.view.target);
        }
        const actorSequence = snapshot.actor_sequence ?? snapshot.sequence;
        if (actorSequence !== appliedActorSequence) {
          const actorStates = snapshot.actors ?? {};
          for (const [actorId, handle] of actors) {
            const state = actorStates[actorId];
            if (state === undefined) continue;
            handle.root.position.set(...state.position_m);
            handle.root.quaternion.set(...state.quaternion_xyzw);
            handle.hasState = true;
            handle.stateVisible = state.visible !== false;
            handle.root.visible = handle.ready && handle.stateVisible;
            if (handle.robot !== null) {
              handle.definition.joint_names.forEach((jointName, index) => {
                const joint = handle.robot.joints[jointName];
                if (joint === undefined) {
                  throw new Error(`browser URDF is missing runtime joint: ${jointName}`);
                }
                joint.setJointValue(state.joint_positions[index]);
              });
            }
          }
          appliedActorSequence = actorSequence;
          if (window.visualizationThreeAssetsReady === true &&
              !sceneAssetFailure && !readyStatusScheduled) {
            readyStatusScheduled = true;
            status.dataset.state = 'ready';
            status.textContent = 'Three.js 彩色场景';
            window.setTimeout(() => status.remove(), 700);
          }
        }
        const trails = snapshot.trails ?? {};
        const trailSequences = snapshot.trail_sequences ?? {};
        for (const [trailId, handle] of trailHandles) {
          const trailSequence = trailSequences[trailId] ?? snapshot.sequence;
          if (trailSequence === handle.appliedSequence) continue;
          updateTrail(handle, trails[trailId] ?? []);
          handle.appliedSequence = trailSequence;
        }
        const polylineOverlays = snapshot.polyline_overlays ?? {};
        const polylineSequences = snapshot.polyline_overlay_sequences ?? {};
        for (const [overlayId, handle] of polylineOverlayHandles) {
          const overlaySequence = polylineSequences[overlayId] ?? snapshot.sequence;
          if (overlaySequence === handle.appliedSequence) continue;
          updatePolylineOverlay(handle, polylineOverlays[overlayId] ?? []);
          handle.appliedSequence = overlaySequence;
        }
      }

      let renderWidth = 0;
      let renderHeight = 0;
      window.visualizationThreeRenderer = {
        drawTo(context, width, height) {
          if (!(width > 0) || !(height > 0)) return false;
          if (renderWidth !== width || renderHeight !== height) {
            renderWidth = width;
            renderHeight = height;
            renderer.setSize(width, height, false);
          }
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          applySnapshot();
          renderer.render(scene, camera);
          context.drawImage(renderer.domElement, 0, 0, width, height);
          return true;
        }
      };
      window.visualizationRequestRedraw?.();
    }

    startThreeRenderer().catch(failThreeRenderer);
  </script>
"""


def _build_button_content(
    buttons: tuple[VisualizationButton, ...],
) -> str:
    return "\n".join(
        (
            '          <button type="button" class="environment-action" '
            f'data-button-id="{escape(button.button_id, quote=True)}" '
            f'data-style="{escape(button.style, quote=True)}">'
            f"{escape(button.label)}</button>"
        )
        for button in buttons
    )


def _build_dashboard_panel(
    panel_id: str,
    title: str,
    body: str,
    *,
    initial_height: int,
    body_class: str = "",
    body_id: str = "",
) -> str:
    class_suffix = f" {body_class}" if body_class else ""
    id_attribute = f' id="{body_id}"' if body_id else ""
    return (
        f'      <section class="dashboard-panel" data-panel-id="{panel_id}" '
        f'style="--panel-height:{initial_height}px">\n'
        '        <header class="dashboard-panel-header" draggable="true">\n'
        f"          <span>{escape(title)}</span>\n"
        '          <button type="button" class="panel-collapse" '
        f'aria-label="折叠{escape(title, quote=True)}" title="折叠/展开">'
        "−</button>\n"
        "        </header>\n"
        f'        <div{id_attribute} class="dashboard-panel-body{class_suffix}">\n'
        f"{body}\n"
        "        </div>\n"
        '        <div class="panel-resize-handle" role="separator" '
        'aria-orientation="horizontal" title="拖动调整高度"></div>\n'
        "      </section>"
    )


def _build_dashboard(
    buttons: tuple[VisualizationButton, ...],
    curves: tuple[VisualizationCurve, ...],
    *,
    show_output: bool,
) -> str:
    panels: list[str] = []
    if buttons:
        panels.append(
            _build_dashboard_panel(
                "buttons",
                "按钮区",
                _build_button_content(buttons),
                initial_height=190,
                body_class="environment-actions",
                body_id="environment-actions",
            )
        )
    if curves:
        legend = "\n".join(
            (
                '            <span class="curve-legend-item" '
                f'data-metric-id="{escape(curve.metric_id, quote=True)}">'
                f'<i style="background:{escape(curve.color, quote=True)}"></i>'
                f"{escape(curve.label)}"
                f"{f' ({escape(curve.unit)})' if curve.unit else ''}</span>"
            )
            for curve in curves
        )
        curve_body = (
            '          <div id="curve-legend" class="curve-legend">\n'
            f"{legend}\n"
            "          </div>\n"
            '          <canvas id="metric-chart" aria-label="指标曲线"></canvas>'
        )
        panels.append(
            _build_dashboard_panel(
                "curves",
                "曲线区",
                curve_body,
                initial_height=270,
                body_class="curve-panel-body",
            )
        )
    if show_output:
        panels.append(
            _build_dashboard_panel(
                "output",
                "输出区",
                '          <div id="dashboard-output" class="dashboard-output" '
                'role="log" aria-live="polite"></div>',
                initial_height=190,
                body_class="output-panel-body",
            )
        )
    if not panels:
        return ""
    rendered_panels = "\n".join(panels)
    return (
        '    <aside id="dashboard" class="dashboard" '
        'aria-label="可视化面板">\n'
        '      <div id="dashboard-width-handle" class="dashboard-width-handle" '
        'role="separator" aria-orientation="vertical" '
        'title="拖动调整侧栏宽度"></div>\n'
        f"{rendered_panels}\n"
        "    </aside>"
    )


def build_browser_page(
    *,
    title: str,
    background: str,
    buttons: Iterable[VisualizationButton] = (),
    curves: Iterable[VisualizationCurve] = (),
    show_output: bool = False,
    scene_style: BrowserSceneStyle | None = None,
    trails: Iterable[VisualizationTrail] = (),
    three_scene: BrowserThreeScene | None = None,
    resolution: tuple[int, int] = (1280, 960),
) -> bytes:
    """Build the self-contained viewer, controls, recorder, and dashboard."""

    frozen_buttons = normalize_buttons(buttons)
    frozen_curves = normalize_curves(curves)
    frozen_trails = normalize_trails(trails)
    if not isinstance(show_output, bool):
        raise TypeError("show_output must be a bool")
    if scene_style is not None and not isinstance(scene_style, BrowserSceneStyle):
        raise TypeError("scene_style must be a BrowserSceneStyle or None")
    if three_scene is not None and not isinstance(three_scene, BrowserThreeScene):
        raise TypeError("three_scene must be a BrowserThreeScene or None")
    if (
        not isinstance(resolution, tuple)
        or len(resolution) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolution
        )
    ):
        raise ValueError("resolution must be a pair of positive integers")
    canvas_width, canvas_height = resolution
    safe_title = escape(title)
    effective_background = (
        scene_style.background_color if scene_style is not None else background
    )
    safe_background = escape(effective_background, quote=True)
    background_json = json.dumps(effective_background, ensure_ascii=True)
    dashboard = _build_dashboard(
        frozen_buttons,
        frozen_curves,
        show_output=show_output,
    )
    curve_definitions_json = json.dumps(
        [
            {
                "metric_id": curve.metric_id,
                "label": curve.label,
                "color": curve.color,
                "unit": curve.unit,
            }
            for curve in frozen_curves
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    curve_definitions_json = (
        curve_definitions_json.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    scene_style_json = json.dumps(
        None if scene_style is None else scene_style.as_payload(),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    trail_definitions_json = json.dumps(
        [trail.as_payload() for trail in frozen_trails],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    three_scene_json = json.dumps(
        None if three_scene is None else three_scene.as_payload(),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    three_scene_json = (
        three_scene_json.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    three_renderer_module = "" if three_scene is None else _THREE_RENDERER_MODULE
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root{{--ui-control-font-size:8px;--ui-detail-font-size:7px;
      --ui-line-height:1.25}}
    html,body{{width:100%;height:100%;margin:0;overflow:hidden}}
    body{{background:{safe_background};font-family:sans-serif;color:#ddd;
      font-size:var(--ui-control-font-size);line-height:var(--ui-line-height)}}
    main{{position:relative;width:100%;height:100%;overflow:hidden;
      touch-action:none;user-select:none;cursor:default;background:{safe_background}}}
    main.camera-dragging{{cursor:grabbing}}
    #frame{{width:100%;height:100%;object-fit:contain;pointer-events:none}}
    .hud{{position:absolute;left:10px;background:rgba(0,0,0,.58);
      pointer-events:none}}
    .help{{bottom:6px;padding:3px 5px;border-radius:3px;
      font-size:var(--ui-detail-font-size);line-height:var(--ui-line-height);color:#aaa}}
    .help span{{display:block;white-space:nowrap}}
    .recorder{{position:absolute;z-index:2;top:8px;left:8px;display:flex;
      flex-direction:column;gap:2px;padding:3px;border-radius:5px;
      background:rgba(0,0,0,.58)}}
    .recorder button{{height:20px;padding:0 5px;border:1px solid #555;
      border-radius:3px;background:#242424;color:#ddd;
      font-size:var(--ui-control-font-size);cursor:pointer}}
    .recorder button:hover:not(:disabled){{background:#343434}}
    .recorder button:disabled{{cursor:default;opacity:.42}}
    .recorder[data-state="recording"] .record-start{{border-color:#d94a4a;
      background:#8d2424;color:#fff;opacity:1}}
    .recorder[data-state="paused"] .record-pause{{border-color:#d39a35;
      background:#654813;color:#fff}}
    .overlay-visibility-controls{{position:absolute;z-index:2;top:8px;left:72px;
      display:flex;flex-direction:column;align-items:stretch;gap:2px;padding:3px;
      border-radius:5px;background:rgba(0,0,0,.58)}}
    .overlay-visibility-controls button{{height:20px;padding:0 6px;
      border:1px solid #526271;border-radius:3px;background:#26333f;color:#dce8f2;
      font-size:var(--ui-control-font-size);cursor:pointer;white-space:nowrap}}
    .overlay-visibility-controls button:hover{{background:#344757}}
    .overlay-visibility-controls button[data-visible="false"]{{opacity:.62;
      border-color:#4a4a4a;background:#242424;color:#b7b7b7}}
    .dashboard{{position:absolute;z-index:3;top:8px;right:8px;width:300px;
      min-width:210px;max-width:min(640px,70vw);height:calc(100% - 16px);
      display:flex;flex-direction:column;gap:5px;box-sizing:border-box;
      padding-left:6px;overflow-y:auto;overflow-x:hidden}}
    .dashboard-width-handle{{position:fixed;z-index:4;width:7px;cursor:ew-resize;
      touch-action:none;border-left:1px solid transparent}}
    .dashboard-width-handle:hover,.dashboard-width-handle.dragging{{
      border-left-color:#5f9ec7;background:rgba(75,133,170,.18)}}
    .dashboard-panel{{flex:0 0 var(--panel-height);min-height:68px;
      display:flex;flex-direction:column;overflow:hidden;border:1px solid #444;
      border-radius:5px;background:rgba(13,13,13,.86);box-shadow:0 2px 8px #0008}}
    .dashboard-panel.dragging{{opacity:.45}}
    .dashboard-panel.drag-over{{border-color:#62a8d5}}
    .dashboard-panel-header{{height:25px;min-height:25px;display:flex;
      align-items:center;justify-content:space-between;padding:0 5px 0 8px;
      background:#262626;color:#ddd;font-size:var(--ui-control-font-size);
      line-height:var(--ui-line-height);font-weight:600;cursor:grab}}
    .dashboard-panel-header:active{{cursor:grabbing}}
    .panel-collapse{{width:20px;height:19px;padding:0;border:1px solid #555;
      border-radius:3px;background:#1d1d1d;color:#bbb;
      font-size:var(--ui-control-font-size);line-height:var(--ui-line-height);cursor:pointer}}
    .dashboard-panel-body{{flex:1;min-height:0;overflow:auto;padding:5px}}
    .panel-resize-handle{{height:5px;min-height:5px;cursor:ns-resize;
      touch-action:none;border-top:1px solid transparent}}
    .panel-resize-handle:hover,.panel-resize-handle.dragging{{
      border-top-color:#5f9ec7;background:rgba(75,133,170,.18)}}
    .dashboard-panel.collapsed{{flex-basis:31px!important;min-height:31px}}
    .dashboard-panel.collapsed .dashboard-panel-body,
    .dashboard-panel.collapsed .panel-resize-handle{{display:none}}
    .environment-actions{{display:flex;flex-direction:column;
      align-items:stretch;gap:3px}}
    .environment-actions button{{min-width:64px;height:22px;padding:0 7px;
      border:1px solid #555;border-radius:3px;background:#242424;color:#ddd;
      font-size:var(--ui-control-font-size);line-height:var(--ui-line-height);
      cursor:pointer}}
    .environment-actions button:hover:not(:disabled){{background:#343434}}
    .environment-actions button:disabled{{cursor:wait;opacity:.55}}
    .environment-actions button[data-style="primary"]{{border-color:#3e7da8;
      background:#1d4f70}}
    .environment-actions button[data-style="danger"]{{border-color:#b94a4a;
      background:#722626}}
    .curve-panel-body{{display:flex;flex-direction:column;overflow:hidden}}
    .curve-legend{{display:flex;flex-wrap:wrap;gap:3px 8px;margin-bottom:3px;
      color:#bbb;font-size:var(--ui-detail-font-size);
      line-height:var(--ui-line-height)}}
    .curve-legend-item{{display:inline-flex;align-items:center;gap:3px}}
    .curve-legend-item i{{display:inline-block;width:8px;height:2px}}
    #metric-chart{{display:block;flex:1;min-height:70px;width:100%;height:100%;
      pointer-events:none;background:#121212;border-radius:3px}}
    .output-panel-body{{padding:0;overflow:hidden}}
    .dashboard-output{{height:100%;box-sizing:border-box;overflow:auto;
      padding:5px 6px;background:#101010;color:#c9c9c9;font-family:monospace;
      font-size:var(--ui-detail-font-size);line-height:1.35;
      user-select:text;white-space:pre-wrap;overflow-wrap:anywhere}}
    .dashboard-log-line{{margin:0 0 2px}}
    .dashboard-log-line[data-level="debug"]{{color:#8b8b8b}}
    .dashboard-log-line[data-level="warning"]{{color:#e1b55b}}
    .dashboard-log-line[data-level="error"]{{color:#ff7777}}
    .three-renderer-status{{position:absolute;z-index:5;top:9px;left:50%;
      transform:translateX(-50%);padding:5px 9px;border:1px solid #47769a;
      border-radius:4px;background:rgba(7,16,27,.88);color:#bce6ff;
      font-size:var(--ui-control-font-size);line-height:var(--ui-line-height);
      pointer-events:none;box-shadow:0 2px 8px #0008}}
    .three-renderer-status[data-state="error"]{{border-color:#b94a4a;
      background:rgba(67,15,20,.92);color:#ffd0d0}}
    .three-scene-legend{{position:absolute;z-index:2;left:8px;bottom:52px;
      display:flex;flex-direction:column;gap:2px;padding:4px 5px;border-radius:4px;
      background:rgba(7,12,18,.62);color:#c8d2dc;font-size:var(--ui-detail-font-size);
      line-height:var(--ui-line-height);pointer-events:none}}
    .three-scene-legend span{{display:flex;align-items:center;gap:4px;white-space:nowrap}}
    .three-scene-legend i{{display:block;width:10px;height:2px;border-radius:1px;
      box-shadow:0 0 3px currentColor}}
  </style>
</head>
<body>
  <main id="viewport">
    <canvas id="frame" width="{canvas_width}" height="{canvas_height}"
      role="img" aria-label="simulation"></canvas>
    <div id="recorder" class="recorder" data-state="idle"
      role="group" aria-label="录制控制">
      <button id="record-start" class="record-start" type="button" disabled>
        ● 录制
      </button>
      <button id="record-pause" class="record-pause" type="button" disabled>
        Ⅱ 暂停
      </button>
      <button id="record-stop" type="button" disabled>■ 停止</button>
    </div>
{dashboard}
    <div class="hud help">
      <span>{escape(CAMERA_HELP)}</span>
      <span>右键拖动：平移</span>
      <span>滚轮：缩放</span>
      <span>R：重置</span>
    </div>
  </main>
  <script>
    'use strict';

    const viewport = document.getElementById('viewport');
    const frame = document.getElementById('frame');
    const frameContext = frame.getContext('2d', {{alpha: false}});
    const recorderControls = document.getElementById('recorder');
    const recordStart = document.getElementById('record-start');
    const recordPause = document.getElementById('record-pause');
    const recordStop = document.getElementById('record-stop');
    const dashboard = document.getElementById('dashboard');
    const dashboardWidthHandle = document.getElementById('dashboard-width-handle');
    const environmentActions = document.getElementById('environment-actions');
    const metricChart = document.getElementById('metric-chart');
    const metricChartContext = metricChart === null ? null :
      metricChart.getContext('2d', {{alpha: false}});
    const dashboardOutput = document.getElementById('dashboard-output');
    const curveDefinitions = Object.freeze({curve_definitions_json});
    const browserSceneStyle = {scene_style_json};
    const trajectoryDefinitions = Object.freeze({trail_definitions_json});
    const browserThreeScene = {three_scene_json};
    const rootComputedStyle = getComputedStyle(document.documentElement);
    const uiFontPixels = Object.freeze({{
      control: Number.parseFloat(
        rootComputedStyle.getPropertyValue('--ui-control-font-size')
      ) || 8,
      detail: Number.parseFloat(
        rootComputedStyle.getPropertyValue('--ui-detail-font-size')
      ) || 7
    }});
    const primaryPointerEvent = 'visualization-primary-pointer';
    if (frameContext === null) throw new Error('2D canvas is unavailable');
    if (metricChart !== null && metricChartContext === null) {{
      throw new Error('dashboard curve canvas is unavailable');
    }}

    let frameSequence = 0;
    let hasFrame = false;
    let latestFrameImage = null;
    let presentationSequence = 0;
    let presentationRenderRevision = 0;
    let controlSequence = 0;
    let defaultCameraView = null;
    let presentationSnapshot = {{
      sequence: 0,
      render_revision: presentationRenderRevision,
      recording_paused: false,
      view: null,
      trails: {{}},
      trail_sequences: {{}},
      polyline_overlays: {{}},
      polyline_overlay_sequences: {{}},
      actors: {{}},
      actor_sequence: 0
    }};
    window.visualizationThreeSceneConfig = browserThreeScene;
    window.visualizationSceneStyle = browserSceneStyle;
    window.visualizationTrajectoryDefinitions = trajectoryDefinitions;
    window.visualizationPresentationSnapshot = presentationSnapshot;

    function vectorSubtract(left, right) {{
      return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
    }}

    function vectorDot(left, right) {{
      return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    }}

    function vectorCross(left, right) {{
      return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]
      ];
    }}

    function vectorNormalize(value) {{
      const length = Math.hypot(value[0], value[1], value[2]);
      if (!(length > 1e-9)) return null;
      return [value[0] / length, value[1] / length, value[2] / length];
    }}

    function cloneCameraView(view) {{
      if (view === null) return null;
      return {{
        eye: [...view.eye],
        target: [...view.target],
        vertical_fov: view.vertical_fov,
        aspect: view.aspect,
        client_sequence: view.client_sequence ?? 0
      }};
    }}

    function previewCameraCommand(command) {{
      if (browserThreeScene === null || presentationSnapshot.view === null) return;
      let view = cloneCameraView(presentationSnapshot.view);
      if (command.reset && defaultCameraView !== null) {{
        view = cloneCameraView(defaultCameraView);
      }}
      let target = [...view.target];
      const offset = vectorSubtract(view.eye, target);
      let distance = Math.hypot(...offset);
      if (!(distance > 1e-9)) return;
      let yaw = Math.atan2(offset[1], offset[0]);
      let pitch = Math.asin(Math.max(-1, Math.min(1, offset[2] / distance)));
      if (command.orbit_x || command.orbit_y) {{
        yaw -= command.orbit_x * 2 * Math.PI;
        yaw = Math.atan2(Math.sin(yaw), Math.cos(yaw));
        pitch += command.orbit_y * 2 * Math.PI;
        const maxPitch = 85 * Math.PI / 180;
        pitch = Math.max(-maxPitch, Math.min(maxPitch, pitch));
      }}
      if (command.zoom) {{
        const exponent = Math.max(-2, Math.min(2, command.zoom * 0.001));
        distance *= Math.exp(exponent);
        const defaultDistance = defaultCameraView === null ? distance : Math.hypot(
          ...vectorSubtract(defaultCameraView.eye, defaultCameraView.target)
        );
        distance = Math.max(
          Math.max(0.1, 0.1 * defaultDistance),
          Math.min(10 * defaultDistance, distance)
        );
      }}
      const cosPitch = Math.cos(pitch);
      let nextOffset = [
        distance * cosPitch * Math.cos(yaw),
        distance * cosPitch * Math.sin(yaw),
        distance * Math.sin(pitch)
      ];
      if (command.pan_x || command.pan_y) {{
        const forward = nextOffset.map(value => -value / distance);
        const rightLength = Math.hypot(forward[0], forward[1]);
        if (!(rightLength > 1e-9)) return;
        const right = [
          forward[1] / rightLength,
          -forward[0] / rightLength,
          0
        ];
        const viewUp = [
          right[1] * forward[2],
          -right[0] * forward[2],
          right[0] * forward[1] - right[1] * forward[0]
        ];
        const scale = 2 * distance * Math.tan(0.5 * view.vertical_fov);
        target = target.map((value, index) => value + scale * (
          -command.pan_x * right[index] + command.pan_y * viewUp[index]
        ));
      }}
      view = {{
        ...view,
        eye: target.map((value, index) => value + nextOffset[index]),
        target,
        client_sequence: command.client_sequence
      }};
      presentationRenderRevision += 1;
      presentationSnapshot = {{
        ...presentationSnapshot,
        render_revision: presentationRenderRevision,
        view
      }};
      window.visualizationPresentationSnapshot = presentationSnapshot;
      window.visualizationRequestRedraw?.();
    }}

    function projectEnvironmentPoint(point) {{
      const view = presentationSnapshot.view;
      if (view === null || !Array.isArray(point) || point.length !== 3) return null;
      const forward = vectorNormalize(vectorSubtract(view.target, view.eye));
      if (forward === null) return null;
      const right = vectorNormalize(vectorCross(forward, [0, 0, 1]));
      if (right === null) return null;
      const up = vectorCross(right, forward);
      const relative = vectorSubtract(point, view.eye);
      const depth = vectorDot(relative, forward);
      if (!(depth > 0.01)) return null;
      const halfHeight = depth * Math.tan(view.vertical_fov * 0.5);
      const halfWidth = halfHeight * view.aspect;
      if (!(halfHeight > 0) || !(halfWidth > 0)) return null;
      const normalizedX = vectorDot(relative, right) / halfWidth;
      const normalizedY = vectorDot(relative, up) / halfHeight;
      return {{
        x: (normalizedX + 1) * 0.5 * frame.width,
        y: (1 - normalizedY) * 0.5 * frame.height,
        depth
      }};
    }}

    function environmentPointDepth(point) {{
      const view = presentationSnapshot.view;
      if (view === null) return Number.NEGATIVE_INFINITY;
      const forward = vectorNormalize(vectorSubtract(view.target, view.eye));
      if (forward === null) return Number.NEGATIVE_INFINITY;
      return vectorDot(vectorSubtract(point, view.eye), forward);
    }}

    function interpolatePoint(start, end, alpha) {{
      return [
        start[0] + (end[0] - start[0]) * alpha,
        start[1] + (end[1] - start[1]) * alpha,
        start[2] + (end[2] - start[2]) * alpha
      ];
    }}

    function drawLightWash(color, x, y, radius, opacity) {{
      const gradient = frameContext.createRadialGradient(x, y, 0, x, y, radius);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, `${{color}}00`);
      frameContext.save();
      frameContext.globalCompositeOperation = 'screen';
      frameContext.globalAlpha = opacity;
      frameContext.fillStyle = gradient;
      frameContext.fillRect(0, 0, frame.width, frame.height);
      frameContext.restore();
    }}

    function drawPresentationLighting() {{
      if (browserSceneStyle === null) return;
      const strength = browserSceneStyle.light_strength;
      const longest = Math.max(frame.width, frame.height);
      drawLightWash(
        browserSceneStyle.key_light_color,
        frame.width * 0.18,
        frame.height * 0.10,
        longest * 0.72,
        strength * 0.46
      );
      drawLightWash(
        browserSceneStyle.fill_light_color,
        frame.width * 0.92,
        frame.height * 0.52,
        longest * 0.68,
        strength * 0.40
      );
      drawLightWash(
        browserSceneStyle.rim_light_color,
        frame.width * 0.58,
        -frame.height * 0.08,
        longest * 0.62,
        strength * 0.34
      );
    }}

    function drawProjectedLine(start, end, color, width, opacity, dash = []) {{
      const nearDepth = 0.011;
      let clippedStart = start;
      let clippedEnd = end;
      let startDepth = environmentPointDepth(clippedStart);
      let endDepth = environmentPointDepth(clippedEnd);
      if (startDepth <= nearDepth && endDepth <= nearDepth) return;
      if (startDepth <= nearDepth) {{
        const alpha = (nearDepth - startDepth) / (endDepth - startDepth);
        clippedStart = interpolatePoint(clippedStart, clippedEnd, alpha);
        startDepth = nearDepth;
      }} else if (endDepth <= nearDepth) {{
        const alpha = (nearDepth - startDepth) / (endDepth - startDepth);
        clippedEnd = interpolatePoint(clippedStart, clippedEnd, alpha);
        endDepth = nearDepth;
      }}
      void startDepth;
      void endDepth;
      const projectedStart = projectEnvironmentPoint(clippedStart);
      const projectedEnd = projectEnvironmentPoint(clippedEnd);
      if (projectedStart === null || projectedEnd === null) return;
      frameContext.save();
      frameContext.globalAlpha = opacity;
      frameContext.strokeStyle = color;
      frameContext.lineWidth = width;
      frameContext.setLineDash(dash);
      frameContext.beginPath();
      frameContext.moveTo(projectedStart.x, projectedStart.y);
      frameContext.lineTo(projectedEnd.x, projectedEnd.y);
      frameContext.stroke();
      frameContext.restore();
    }}

    function drawColoredGridFloor() {{
      if (browserSceneStyle === null || presentationSnapshot.view === null) return;
      const divisions = browserSceneStyle.grid_divisions;
      const halfSize = browserSceneStyle.grid_size_m * 0.5;
      const spacing = browserSceneStyle.grid_size_m / divisions;
      const z = browserSceneStyle.grid_plane_z_m;

      frameContext.save();
      frameContext.globalCompositeOperation = 'screen';
      for (let index = 0; index <= divisions; index += 1) {{
        const offset = -halfSize + index * spacing;
        const centered = index * 2 === divisions;
        const major = centered || index % 4 === 0;
        drawProjectedLine(
          [offset, -halfSize, z],
          [offset, halfSize, z],
          centered ? browserSceneStyle.y_axis_color :
            major ? browserSceneStyle.grid_major_color :
            browserSceneStyle.grid_minor_color,
          centered ? 1.8 : major ? 1.25 : 0.75,
          centered ? 0.82 : major ? 0.56 : 0.38,
          centered || major ? [] : [2, 4]
        );
        drawProjectedLine(
          [-halfSize, offset, z],
          [halfSize, offset, z],
          centered ? browserSceneStyle.x_axis_color :
            major ? browserSceneStyle.grid_major_color :
            browserSceneStyle.grid_minor_color,
          centered ? 1.8 : major ? 1.25 : 0.75,
          centered ? 0.82 : major ? 0.56 : 0.38,
          centered || major ? [] : [2, 4]
        );
      }}
      frameContext.restore();
    }}

    function drawTrajectoryMarkers() {{
      if (!trajectoryDefinitions.length) return;
      const trails = presentationSnapshot.trails || {{}};
      for (const definition of trajectoryDefinitions) {{
        const points = Array.isArray(trails[definition.trail_id]) ?
          trails[definition.trail_id] : [];
        const projected = points.map(projectEnvironmentPoint).filter(point => point !== null);
        if (!projected.length) continue;

        frameContext.save();
        frameContext.globalCompositeOperation = 'screen';
        frameContext.strokeStyle = definition.color;
        frameContext.lineWidth = definition.line_width_px;
        frameContext.lineJoin = 'round';
        frameContext.lineCap = 'round';
        frameContext.shadowColor = definition.color;
        frameContext.shadowBlur = definition.line_width_px * 2.6;
        frameContext.globalAlpha = 0.92;
        if (projected.length >= 2) {{
          frameContext.beginPath();
          frameContext.moveTo(projected[0].x, projected[0].y);
          for (let index = 1; index < projected.length; index += 1) {{
            frameContext.lineTo(projected[index].x, projected[index].y);
          }}
          frameContext.stroke();
        }}

        const markerStride = Math.max(1, Math.floor(projected.length / 18));
        frameContext.shadowBlur = 0;
        frameContext.fillStyle = definition.color;
        frameContext.globalAlpha = 0.58;
        for (let index = 0; index < projected.length - 1; index += markerStride) {{
          frameContext.beginPath();
          frameContext.arc(projected[index].x, projected[index].y, 1.8, 0, Math.PI * 2);
          frameContext.fill();
        }}

        const endpoint = projected[projected.length - 1];
        frameContext.globalAlpha = 1;
        frameContext.shadowColor = definition.color;
        frameContext.shadowBlur = definition.marker_radius_px * 2.5;
        frameContext.beginPath();
        frameContext.arc(
          endpoint.x,
          endpoint.y,
          definition.marker_radius_px,
          0,
          Math.PI * 2
        );
        frameContext.fill();
        frameContext.shadowBlur = 0;
        frameContext.font = `600 ${{uiFontPixels.control}}px sans-serif`;
        frameContext.fillText(
          definition.label,
          endpoint.x + definition.marker_radius_px + 5,
          endpoint.y - definition.marker_radius_px - 2
        );
        frameContext.restore();
      }}
    }}

    function drawCompositedFrame() {{
      if (frame.width <= 0 || frame.height <= 0) return;
      frameContext.save();
      frameContext.setTransform(1, 0, 0, 1, 0, 0);
      frameContext.globalCompositeOperation = 'source-over';
      frameContext.globalAlpha = 1;
      frameContext.filter = 'none';
      frameContext.fillStyle = browserSceneStyle === null ?
        {background_json} : browserSceneStyle.background_color;
      frameContext.fillRect(0, 0, frame.width, frame.height);

      if (browserThreeScene !== null &&
          window.visualizationThreeRenderer !== undefined) {{
        try {{
          if (window.visualizationThreeRenderer.drawTo(
            frameContext,
            frame.width,
            frame.height
          )) {{
            markFrameReady();
            frameContext.restore();
            return;
          }}
        }} catch (error) {{
          console.error('Three.js scene render failed', error);
        }}
      }}

      if (latestFrameImage === null) {{
        frameContext.restore();
        return;
      }}
      if (browserSceneStyle === null) {{
        frameContext.drawImage(latestFrameImage, 0, 0, frame.width, frame.height);
      }} else {{
        const strength = browserSceneStyle.color_grade_strength;
        frameContext.filter = `contrast(${{1 + strength * 0.52}}) ` +
          `saturate(${{1 + strength * 0.90}}) brightness(${{1 - strength * 0.10}})`;
        frameContext.drawImage(latestFrameImage, 0, 0, frame.width, frame.height);
        frameContext.filter = 'none';

        const depthTint = frameContext.createLinearGradient(0, 0, 0, frame.height);
        depthTint.addColorStop(0, browserSceneStyle.background_color);
        depthTint.addColorStop(1, browserSceneStyle.floor_tint);
        frameContext.globalCompositeOperation = 'multiply';
        frameContext.globalAlpha = strength * 0.62;
        frameContext.fillStyle = depthTint;
        frameContext.fillRect(0, 0, frame.width, frame.height);

        const materialTint = frameContext.createLinearGradient(0, 0, frame.width, 0);
        materialTint.addColorStop(0, browserSceneStyle.key_light_color);
        materialTint.addColorStop(0.52, browserSceneStyle.rim_light_color);
        materialTint.addColorStop(1, browserSceneStyle.fill_light_color);
        frameContext.globalCompositeOperation = 'soft-light';
        frameContext.globalAlpha = strength * 0.34;
        frameContext.fillStyle = materialTint;
        frameContext.fillRect(0, 0, frame.width, frame.height);
        drawPresentationLighting();
      }}
      frameContext.restore();
      drawColoredGridFloor();
      drawTrajectoryMarkers();
    }}
    let compositedFrameScheduled = false;
    function requestCompositedFrame() {{
      if (compositedFrameScheduled) return;
      compositedFrameScheduled = true;
      requestAnimationFrame(() => {{
        compositedFrameScheduled = false;
        drawCompositedFrame();
      }});
    }}
    window.visualizationRequestRedraw = requestCompositedFrame;

    const emptyCommand = () => ({{
      orbit_x: 0, orbit_y: 0, pan_x: 0, pan_y: 0, zoom: 0, reset: false,
      client_sequence: 0
    }});
    const hasCommand = command => command.reset || command.orbit_x ||
      command.orbit_y || command.pan_x || command.pan_y || command.zoom ||
      command.client_sequence;

    let pending = emptyCommand();
    let scheduled = false;
    let fallbackSending = false;
    let controlSocket = null;
    let reconnectTimer = null;

    function takePending() {{
      const command = pending;
      pending = emptyCommand();
      return command;
    }}

    function schedule() {{
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(flushControls);
    }}

    function queueMotion(values) {{
      controlSequence += 1;
      for (const [key, value] of Object.entries(values)) pending[key] += value;
      pending.client_sequence = controlSequence;
      previewCameraCommand({{...values, reset: false, client_sequence: controlSequence}});
      schedule();
    }}

    function queueReset() {{
      controlSequence += 1;
      pending = emptyCommand();
      pending.reset = true;
      pending.client_sequence = controlSequence;
      previewCameraCommand({{...pending}});
      schedule();
    }}

    function connectControls() {{
      const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
      const socket = new WebSocket(`${{scheme}}://${{location.host}}/control`);
      controlSocket = socket;
      socket.addEventListener('open', schedule);
      socket.addEventListener('error', () => socket.close());
      socket.addEventListener('close', () => {{
        if (controlSocket === socket) controlSocket = null;
        if (reconnectTimer === null) {{
          reconnectTimer = setTimeout(() => {{
            reconnectTimer = null;
            connectControls();
          }}, 500);
        }}
      }});
    }}

    function flushHttpFallback() {{
      if (fallbackSending || !hasCommand(pending)) return;
      const command = takePending();
      fallbackSending = true;
      fetch('/camera', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(command),
        cache: 'no-store'
      }}).catch(error => console.error('camera control failed', error))
        .finally(() => {{
          fallbackSending = false;
          if (hasCommand(pending)) schedule();
        }});
    }}

    function flushControls() {{
      scheduled = false;
      if (!hasCommand(pending)) return;
      if (controlSocket && controlSocket.readyState === WebSocket.OPEN) {{
        if (controlSocket.bufferedAmount > 1024) {{
          setTimeout(schedule, 8);
          return;
        }}
        controlSocket.send(JSON.stringify(takePending()));
        if (hasCommand(pending)) schedule();
        return;
      }}
      flushHttpFallback();
    }}

    function displayedImageRect() {{
      const viewportRect = viewport.getBoundingClientRect();
      if ((!hasFrame && browserThreeScene === null) ||
          frame.width <= 0 || frame.height <= 0) {{
        return {{
          left: viewportRect.left,
          top: viewportRect.top,
          width: Math.max(1, viewportRect.width),
          height: Math.max(1, viewportRect.height)
        }};
      }}
      const scale = Math.min(
        viewportRect.width / frame.width,
        viewportRect.height / frame.height
      );
      const width = Math.max(1, frame.width * scale);
      const height = Math.max(1, frame.height * scale);
      return {{
        left: viewportRect.left + (viewportRect.width - width) * 0.5,
        top: viewportRect.top + (viewportRect.height - height) * 0.5,
        width,
        height
      }};
    }}

    function displayedImageHeight() {{
      return displayedImageRect().height;
    }}

    function emitPrimaryPointer(phase, event, dx = 0, dy = 0) {{
      const rect = displayedImageRect();
      const imageX = (event.clientX - rect.left) / rect.width;
      const imageY = (event.clientY - rect.top) / rect.height;
      const insideImage = imageX >= 0 && imageX <= 1 && imageY >= 0 && imageY <= 1;
      viewport.dispatchEvent(new CustomEvent(primaryPointerEvent, {{
        detail: {{
          phase,
          pointerId: event.pointerId,
          pointerType: event.pointerType,
          frameSequence,
          imageX: insideImage ? imageX : null,
          imageY: insideImage ? imageY : null,
          deltaX: dx / rect.height,
          deltaY: dy / rect.height,
          shiftKey: Boolean(event.shiftKey),
          ctrlKey: Boolean(event.ctrlKey),
          altKey: Boolean(event.altKey),
          metaKey: Boolean(event.metaKey)
        }}
      }}));
    }}

    window.visualizationInteractions = Object.freeze({{
      primaryPointerEvent,
      addPrimaryPointerListener(listener) {{
        viewport.addEventListener(primaryPointerEvent, listener);
        return () => viewport.removeEventListener(primaryPointerEvent, listener);
      }}
    }});

    let environmentActionQueue = Promise.resolve();

    function queueEnvironmentAction(button) {{
      const buttonId = button.dataset.buttonId;
      if (!buttonId || button.disabled) return;
      button.disabled = true;
      environmentActionQueue = environmentActionQueue
        .then(async () => {{
          const response = await fetch('/button', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{button_id: buttonId}}),
            cache: 'no-store'
          }});
          if (!response.ok) {{
            throw new Error(`${{response.status}} ${{response.statusText}}`);
          }}
        }})
        .catch(error => {{
          console.error(`environment action ${{buttonId}} failed`, error);
        }})
        .finally(() => {{
          button.disabled = false;
        }});
    }}

    function environmentActionFromEvent(event) {{
      if (!(event.target instanceof Element)) return;
      const button = event.target.closest('button[data-button-id]');
      if (button !== null) queueEnvironmentAction(button);
    }}

    if (environmentActions !== null) {{
      environmentActions.addEventListener('pointerdown', event => {{
        event.stopPropagation();
        if (event.button !== 0) return;
        event.preventDefault();
        environmentActionFromEvent(event);
      }});
      environmentActions.addEventListener('wheel', event => event.stopPropagation());
      environmentActions.addEventListener('click', event => {{
        // Pointer activation is already sent on press; retain keyboard click.
        if (event.detail === 0) environmentActionFromEvent(event);
      }});
    }}

    const dashboardLayoutKey = 'isaac-visualization.dashboard.v1';
    const dashboardPanels = dashboard === null ? [] :
      Array.from(dashboard.querySelectorAll('.dashboard-panel'));
    let layoutSaveTimer = null;

    function panelHeight(panel) {{
      const configured = Number.parseFloat(
        panel.style.getPropertyValue('--panel-height')
      );
      return Number.isFinite(configured) ? configured : panel.getBoundingClientRect().height;
    }}

    function saveDashboardLayout() {{
      if (dashboard === null) return;
      const state = {{
        version: 1,
        width: dashboard.getBoundingClientRect().width,
        panels: Array.from(dashboard.querySelectorAll('.dashboard-panel')).map(panel => ({{
          id: panel.dataset.panelId,
          height: panelHeight(panel),
          collapsed: panel.classList.contains('collapsed')
        }}))
      }};
      try {{
        localStorage.setItem(dashboardLayoutKey, JSON.stringify(state));
      }} catch (error) {{
        console.debug('dashboard layout could not be saved', error);
      }}
    }}

    function scheduleDashboardLayoutSave() {{
      if (layoutSaveTimer !== null) clearTimeout(layoutSaveTimer);
      layoutSaveTimer = setTimeout(() => {{
        layoutSaveTimer = null;
        saveDashboardLayout();
      }}, 100);
    }}

    function setPanelCollapsed(panel, collapsed) {{
      panel.classList.toggle('collapsed', collapsed);
      const button = panel.querySelector('.panel-collapse');
      if (button !== null) {{
        button.textContent = collapsed ? '+' : '−';
        button.setAttribute('aria-expanded', String(!collapsed));
      }}
    }}

    function restoreDashboardLayout() {{
      if (dashboard === null) return;
      let state = null;
      try {{
        state = JSON.parse(localStorage.getItem(dashboardLayoutKey) || 'null');
      }} catch (error) {{
        console.debug('dashboard layout could not be restored', error);
      }}
      if (state === null || state.version !== 1 || !Array.isArray(state.panels)) return;
      if (Number.isFinite(state.width)) {{
        const width = Math.max(210, Math.min(state.width, Math.min(640, innerWidth * 0.7)));
        dashboard.style.width = `${{width}}px`;
      }}
      const available = new Map(dashboardPanels.map(panel => [panel.dataset.panelId, panel]));
      for (const saved of state.panels) {{
        if (saved === null || typeof saved !== 'object') continue;
        const panel = available.get(saved.id);
        if (panel === undefined) continue;
        if (Number.isFinite(saved.height)) {{
          const height = Math.max(68, Math.min(saved.height, Math.max(90, innerHeight - 48)));
          panel.style.setProperty('--panel-height', `${{height}}px`);
        }}
        setPanelCollapsed(panel, Boolean(saved.collapsed));
        dashboard.appendChild(panel);
        available.delete(saved.id);
      }}
      for (const panel of available.values()) dashboard.appendChild(panel);
    }}

    function positionDashboardWidthHandle() {{
      if (dashboard === null || dashboardWidthHandle === null) return;
      const rect = dashboard.getBoundingClientRect();
      dashboardWidthHandle.style.left = `${{rect.left}}px`;
      dashboardWidthHandle.style.top = `${{rect.top}}px`;
      dashboardWidthHandle.style.height = `${{rect.height}}px`;
    }}

    if (dashboard !== null) {{
      restoreDashboardLayout();
      positionDashboardWidthHandle();
      for (const eventName of ['pointerdown', 'wheel', 'contextmenu']) {{
        dashboard.addEventListener(eventName, event => event.stopPropagation());
      }}
      dashboard.addEventListener('click', event => {{
        if (!(event.target instanceof Element)) return;
        const collapse = event.target.closest('.panel-collapse');
        if (collapse === null) return;
        const panel = collapse.closest('.dashboard-panel');
        if (panel === null) return;
        setPanelCollapsed(panel, !panel.classList.contains('collapsed'));
        scheduleCurveDraw();
        scheduleDashboardLayoutSave();
      }});

      let draggedPanel = null;
      for (const panel of dashboardPanels) {{
        const header = panel.querySelector('.dashboard-panel-header');
        if (header !== null) {{
          header.addEventListener('dragstart', event => {{
            if (event.target instanceof Element &&
                event.target.closest('.panel-collapse') !== null) {{
              event.preventDefault();
              return;
            }}
            draggedPanel = panel;
            panel.classList.add('dragging');
            if (event.dataTransfer !== null) {{
              event.dataTransfer.effectAllowed = 'move';
              event.dataTransfer.setData('text/plain', panel.dataset.panelId || 'panel');
            }}
          }});
          header.addEventListener('dragend', () => {{
            panel.classList.remove('dragging');
            for (const candidate of dashboardPanels) candidate.classList.remove('drag-over');
            draggedPanel = null;
            scheduleDashboardLayoutSave();
          }});
        }}
        panel.addEventListener('dragover', event => {{
          if (draggedPanel === null || draggedPanel === panel) return;
          event.preventDefault();
          panel.classList.add('drag-over');
        }});
        panel.addEventListener('dragleave', () => panel.classList.remove('drag-over'));
        panel.addEventListener('drop', event => {{
          if (draggedPanel === null || draggedPanel === panel) return;
          event.preventDefault();
          panel.classList.remove('drag-over');
          const rect = panel.getBoundingClientRect();
          const before = event.clientY < rect.top + rect.height * 0.5;
          dashboard.insertBefore(draggedPanel, before ? panel : panel.nextSibling);
          scheduleDashboardLayoutSave();
        }});

        const resizeHandle = panel.querySelector('.panel-resize-handle');
        if (resizeHandle !== null) {{
          resizeHandle.addEventListener('pointerdown', event => {{
            if (panel.classList.contains('collapsed')) return;
            event.preventDefault();
            const startY = event.clientY;
            const startHeight = panel.getBoundingClientRect().height;
            resizeHandle.classList.add('dragging');
            resizeHandle.setPointerCapture(event.pointerId);
            const move = moveEvent => {{
              if (moveEvent.pointerId !== event.pointerId) return;
              const height = Math.max(
                68,
                Math.min(startHeight + moveEvent.clientY - startY, Math.max(90, innerHeight - 48))
              );
              panel.style.setProperty('--panel-height', `${{height}}px`);
            }};
            const finish = finishEvent => {{
              if (finishEvent.pointerId !== event.pointerId) return;
              resizeHandle.removeEventListener('pointermove', move);
              resizeHandle.removeEventListener('pointerup', finish);
              resizeHandle.removeEventListener('pointercancel', finish);
              resizeHandle.classList.remove('dragging');
              scheduleDashboardLayoutSave();
            }};
            resizeHandle.addEventListener('pointermove', move);
            resizeHandle.addEventListener('pointerup', finish);
            resizeHandle.addEventListener('pointercancel', finish);
          }});
        }}
      }}

      if (dashboardWidthHandle !== null) {{
        dashboardWidthHandle.addEventListener('pointerdown', event => {{
          event.preventDefault();
          const startX = event.clientX;
          const startWidth = dashboard.getBoundingClientRect().width;
          dashboardWidthHandle.classList.add('dragging');
          dashboardWidthHandle.setPointerCapture(event.pointerId);
          const move = moveEvent => {{
            if (moveEvent.pointerId !== event.pointerId) return;
            const width = Math.max(
              210,
              Math.min(startWidth + startX - moveEvent.clientX, Math.min(640, innerWidth * 0.7))
            );
            dashboard.style.width = `${{width}}px`;
            positionDashboardWidthHandle();
          }};
          const finish = finishEvent => {{
            if (finishEvent.pointerId !== event.pointerId) return;
            dashboardWidthHandle.removeEventListener('pointermove', move);
            dashboardWidthHandle.removeEventListener('pointerup', finish);
            dashboardWidthHandle.removeEventListener('pointercancel', finish);
            dashboardWidthHandle.classList.remove('dragging');
            scheduleDashboardLayoutSave();
          }};
          dashboardWidthHandle.addEventListener('pointermove', move);
          dashboardWidthHandle.addEventListener('pointerup', finish);
          dashboardWidthHandle.addEventListener('pointercancel', finish);
        }});
      }}
      window.addEventListener('resize', () => {{
        const width = Math.min(dashboard.getBoundingClientRect().width, Math.min(640, innerWidth * 0.7));
        dashboard.style.width = `${{Math.max(210, width)}}px`;
        positionDashboardWidthHandle();
      }});
    }}

    const maxCurvePoints = 1500;
    const maxLogLines = 1000;
    const curveHistory = new Map(
      curveDefinitions.map(definition => [definition.metric_id, []])
    );
    let dashboardSequence = 0;
    let curveDrawScheduled = false;

    function scheduleCurveDraw() {{
      if (metricChart === null || curveDrawScheduled) return;
      curveDrawScheduled = true;
      requestAnimationFrame(() => {{
        curveDrawScheduled = false;
        drawMetricChart();
      }});
    }}

    function drawMetricChart() {{
      if (metricChart === null || metricChartContext === null) return;
      const rect = metricChart.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return;
      const pixelRatio = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
      const pixelWidth = Math.max(1, Math.round(rect.width * pixelRatio));
      const pixelHeight = Math.max(1, Math.round(rect.height * pixelRatio));
      if (metricChart.width !== pixelWidth || metricChart.height !== pixelHeight) {{
        metricChart.width = pixelWidth;
        metricChart.height = pixelHeight;
      }}
      const context = metricChartContext;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      const width = rect.width;
      const height = rect.height;
      context.fillStyle = '#121212';
      context.fillRect(0, 0, width, height);

      const allPoints = [];
      for (const points of curveHistory.values()) allPoints.push(...points);
      context.font = `${{uiFontPixels.detail}}px monospace`;
      if (allPoints.length === 0) {{
        context.fillStyle = '#777';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText('等待指标数据…', width * 0.5, height * 0.5);
        return;
      }}

      let minTime = Math.min(...allPoints.map(point => point.time));
      let maxTime = Math.max(...allPoints.map(point => point.time));
      let minValue = Math.min(...allPoints.map(point => point.value));
      let maxValue = Math.max(...allPoints.map(point => point.value));
      if (!(maxTime > minTime)) {{
        minTime -= 0.5;
        maxTime += 0.5;
      }}
      if (!(maxValue > minValue)) {{
        const padding = Math.max(0.01, Math.abs(minValue) * 0.05);
        minValue -= padding;
        maxValue += padding;
      }} else {{
        const padding = (maxValue - minValue) * 0.08;
        minValue -= padding;
        maxValue += padding;
      }}

      const margin = {{left: 48, right: 8, top: 8, bottom: 20}};
      const plotWidth = Math.max(1, width - margin.left - margin.right);
      const plotHeight = Math.max(1, height - margin.top - margin.bottom);
      const xFor = time => margin.left +
        (time - minTime) / (maxTime - minTime) * plotWidth;
      const yFor = value => margin.top +
        (maxValue - value) / (maxValue - minValue) * plotHeight;

      context.lineWidth = 1;
      context.strokeStyle = '#303030';
      context.fillStyle = '#898989';
      context.textBaseline = 'middle';
      for (let index = 0; index <= 4; index += 1) {{
        const alpha = index / 4;
        const y = margin.top + alpha * plotHeight;
        context.beginPath();
        context.moveTo(margin.left, y);
        context.lineTo(margin.left + plotWidth, y);
        context.stroke();
        const value = maxValue - alpha * (maxValue - minValue);
        context.textAlign = 'right';
        context.fillText(value.toFixed(3), margin.left - 4, y);
      }}
      context.textBaseline = 'top';
      context.textAlign = 'left';
      context.fillText(`${{minTime.toFixed(1)}}s`, margin.left, margin.top + plotHeight + 4);
      context.textAlign = 'right';
      context.fillText(
        `${{maxTime.toFixed(1)}}s`,
        margin.left + plotWidth,
        margin.top + plotHeight + 4
      );

      for (const definition of curveDefinitions) {{
        const points = curveHistory.get(definition.metric_id) || [];
        if (points.length === 0) continue;
        context.beginPath();
        points.forEach((point, index) => {{
          const x = xFor(point.time);
          const y = yFor(point.value);
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        }});
        context.strokeStyle = definition.color;
        context.lineWidth = 1.5;
        context.stroke();
      }}
    }}

    function appendDashboardLog(event) {{
      if (dashboardOutput === null) return;
      const stickToBottom = dashboardOutput.scrollHeight - dashboardOutput.scrollTop -
        dashboardOutput.clientHeight < 12;
      const line = document.createElement('div');
      line.className = 'dashboard-log-line';
      line.dataset.level = event.level || 'info';
      const time = Number.isFinite(event.time_s) ? `+${{event.time_s.toFixed(2)}}s ` : '';
      const source = event.source ? `[${{event.source}}] ` : '';
      line.textContent = `${{time}}${{source}}${{event.message}}`;
      dashboardOutput.appendChild(line);
      while (dashboardOutput.childElementCount > maxLogLines) {{
        dashboardOutput.firstElementChild?.remove();
      }}
      if (stickToBottom) dashboardOutput.scrollTop = dashboardOutput.scrollHeight;
    }}

    function consumeDashboardEvent(event) {{
      if (event === null || typeof event !== 'object') return;
      if (event.type === 'metrics' && Number.isFinite(event.time_s) &&
          event.values !== null && typeof event.values === 'object') {{
        for (const definition of curveDefinitions) {{
          const value = event.values[definition.metric_id];
          if (!Number.isFinite(value)) continue;
          const points = curveHistory.get(definition.metric_id);
          points.push({{time: event.time_s, value}});
          if (points.length > maxCurvePoints) points.splice(0, points.length - maxCurvePoints);
        }}
        scheduleCurveDraw();
      }} else if (event.type === 'log' && typeof event.message === 'string') {{
        appendDashboardLog(event);
      }}
    }}

    async function receiveDashboardEvents() {{
      if (curveDefinitions.length === 0 && dashboardOutput === null) return;
      while (true) {{
        try {{
          const response = await fetch(`/dashboard?after=${{dashboardSequence}}`, {{
            cache: 'no-store'
          }});
          if (response.status === 204) {{
            const nextSequence = Number(response.headers.get('X-Dashboard-Sequence'));
            if (Number.isSafeInteger(nextSequence) && nextSequence >= dashboardSequence) {{
              dashboardSequence = nextSequence;
            }}
            continue;
          }}
          if (!response.ok) throw new Error(response.statusText);
          const batch = await response.json();
          if (Number.isSafeInteger(batch.dropped) && batch.dropped > 0) {{
            appendDashboardLog({{
              level: 'warning',
              source: 'dashboard',
              message: `${{batch.dropped}} 条过旧遥测事件已丢弃`
            }});
          }}
          if (Array.isArray(batch.events)) {{
            for (const event of batch.events) consumeDashboardEvent(event);
          }}
          if (!Number.isSafeInteger(batch.sequence) || batch.sequence < dashboardSequence) {{
            throw new Error('invalid dashboard sequence');
          }}
          dashboardSequence = batch.sequence;
        }} catch (error) {{
          console.error('dashboard receive failed', error);
          await new Promise(resolve => setTimeout(resolve, 250));
        }}
      }}
    }}

    async function receivePresentationState() {{
      if (browserSceneStyle === null && !trajectoryDefinitions.length &&
          browserThreeScene === null) return;
      while (true) {{
        try {{
          const response = await fetch(
            `/presentation?after=${{presentationSequence}}`,
            {{cache: 'no-store'}}
          );
          if (response.status === 204) {{
            const current = Number(response.headers.get('X-Presentation-Sequence'));
            if (Number.isSafeInteger(current) && current >= presentationSequence) {{
              presentationSequence = current;
            }}
            continue;
          }}
          if (!response.ok) throw new Error(response.statusText);
          const snapshot = await response.json();
          if (!Number.isSafeInteger(snapshot.sequence) ||
              snapshot.sequence < presentationSequence) {{
            throw new Error('invalid presentation sequence');
          }}
          if (typeof snapshot.recording_paused !== 'boolean') {{
            throw new Error('invalid presentation recording gate');
          }}
          if (snapshot.view !== null && typeof snapshot.view !== 'object') {{
            throw new Error('invalid presentation camera view');
          }}
          if (typeof snapshot.trails !== 'object' || snapshot.trails === null) {{
            throw new Error('invalid presentation trails');
          }}
          if (typeof snapshot.actors !== 'object' || snapshot.actors === null) {{
            throw new Error('invalid presentation actors');
          }}
          if (typeof snapshot.polyline_overlays !== 'object' ||
              snapshot.polyline_overlays === null ||
              typeof snapshot.polyline_overlay_sequences !== 'object' ||
              snapshot.polyline_overlay_sequences === null) {{
            throw new Error('invalid presentation polyline overlays');
          }}
          if (typeof snapshot.trail_sequences !== 'object' ||
              snapshot.trail_sequences === null ||
              !Number.isSafeInteger(snapshot.actor_sequence)) {{
            throw new Error('invalid presentation component sequences');
          }}
          presentationSequence = snapshot.sequence;
          const acknowledgedControlSequence = Number.isSafeInteger(
            snapshot.view?.client_sequence
          ) ? snapshot.view.client_sequence : 0;
          if (defaultCameraView === null && snapshot.view !== null) {{
            defaultCameraView = cloneCameraView(snapshot.view);
          }}
          controlSequence = Math.max(controlSequence, acknowledgedControlSequence);
          const keepLocalPreview = browserThreeScene !== null &&
            presentationSnapshot.view !== null &&
            acknowledgedControlSequence < controlSequence;
          presentationRenderRevision += 1;
          presentationSnapshot = {{
            sequence: snapshot.sequence,
            render_revision: presentationRenderRevision,
            recording_paused: snapshot.recording_paused,
            view: keepLocalPreview ? presentationSnapshot.view : snapshot.view,
            trails: snapshot.trails,
            trail_sequences: snapshot.trail_sequences,
            polyline_overlays: snapshot.polyline_overlays,
            polyline_overlay_sequences: snapshot.polyline_overlay_sequences,
            actors: snapshot.actors,
            actor_sequence: snapshot.actor_sequence
          }};
          window.visualizationPresentationSnapshot = presentationSnapshot;
          window.visualizationApplyRecordingGate?.(snapshot.recording_paused);
          requestCompositedFrame();
        }} catch (error) {{
          console.error('presentation receive failed', error);
          await new Promise(resolve => setTimeout(resolve, 250));
        }}
      }}
    }}

    if (metricChart !== null) {{
      if (typeof ResizeObserver !== 'undefined') {{
        new ResizeObserver(scheduleCurveDraw).observe(metricChart);
      }} else {{
        window.addEventListener('resize', scheduleCurveDraw);
      }}
      scheduleCurveDraw();
    }}

    let activePointer = null;
    let dragMode = null;
    let requiredButton = 0;
    const cameraDragModes = {json.dumps(dict(CAMERA_DRAG_MODES))};
    let lastX = 0;
    let lastY = 0;
    let lastPointerType = '';

    function clearDrag() {{
      const pointerId = activePointer;
      activePointer = null;
      dragMode = null;
      viewport.classList.remove('camera-dragging');
      if (pointerId !== null && viewport.hasPointerCapture(pointerId)) {{
        viewport.releasePointerCapture(pointerId);
      }}
    }}

    function finishDrag(phase, event) {{
      if (dragMode === 'primary') emitPrimaryPointer(phase, event);
      clearDrag();
    }}

    viewport.addEventListener('pointerdown', event => {{
      if (activePointer !== null || recorderControls.contains(event.target) ||
          (dashboard !== null && dashboard.contains(event.target))) return;
      // Alt-left preserves optional tool picking; ordinary left drag rotates.
      dragMode = event.button === 0 && event.altKey ? 'primary' :
        cameraDragModes[String(event.button)];
      if (!dragMode) return;
      requiredButton = [1, 4, 2][event.button];
      event.preventDefault();
      activePointer = event.pointerId;
      lastPointerType = event.pointerType;
      lastX = event.clientX;
      lastY = event.clientY;
      viewport.setPointerCapture(activePointer);
      if (dragMode === 'primary') emitPrimaryPointer('start', event);
      else viewport.classList.add('camera-dragging');
    }});

    viewport.addEventListener('pointermove', event => {{
      if (event.pointerId !== activePointer || dragMode === null) return;
      if (!(event.buttons & requiredButton)) {{
        finishDrag('end', event);
        return;
      }}
      const samples = event.getCoalescedEvents ? event.getCoalescedEvents() : [];
      const current = samples.length ? samples[samples.length - 1] : event;
      const dx = current.clientX - lastX;
      const dy = current.clientY - lastY;
      lastX = current.clientX;
      lastY = current.clientY;
      if (dragMode === 'primary') {{
        emitPrimaryPointer('move', current, dx, dy);
        return;
      }}
      const imageHeight = displayedImageHeight();
      if (dragMode === 'orbit') {{
        queueMotion({{orbit_x: dx / imageHeight, orbit_y: dy / imageHeight}});
      }} else {{
        queueMotion({{pan_x: dx / imageHeight, pan_y: dy / imageHeight}});
      }}
    }});

    for (const eventName of ['pointerup', 'pointercancel', 'lostpointercapture']) {{
      viewport.addEventListener(eventName, event => {{
        if (event.pointerId !== activePointer) return;
        finishDrag(eventName === 'pointerup' ? 'end' : 'cancel', event);
      }});
    }}
    window.addEventListener('blur', () => {{
      if (activePointer === null) return;
      finishDrag('cancel', {{
        pointerId: activePointer,
        pointerType: lastPointerType,
        clientX: lastX,
        clientY: lastY,
        shiftKey: false,
        ctrlKey: false,
        altKey: false,
        metaKey: false
      }});
    }});
    viewport.addEventListener('auxclick', event => {{
      if (event.button === 1) event.preventDefault();
    }});
    viewport.addEventListener('contextmenu', event => event.preventDefault());
    viewport.addEventListener('wheel', event => {{
      event.preventDefault();
      const unit = event.deltaMode === 1 ? 16 :
        event.deltaMode === 2 ? displayedImageHeight() : 1;
      const delta = Math.max(-200, Math.min(200, event.deltaY * unit));
      queueMotion({{zoom: delta}});
    }}, {{passive: false}});
    window.addEventListener('keydown', event => {{
      if (event.key.toLowerCase() === 'r' && !event.repeat) queueReset();
    }});

    const recordingMimeCandidates = [
      'video/mp4;codecs=avc1.42E01E',
      'video/mp4;codecs=avc1',
      'video/mp4',
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm'
    ];
    const recordingSupported = typeof MediaRecorder !== 'undefined' &&
      typeof frame.captureStream === 'function';
    let recordingState = 'idle';
    let mediaRecorder = null;
    let recordingStream = null;
    let recordingChunks = [];
    let recordingGatePaused = presentationSnapshot.recording_paused;
    let recordingAutomaticallyPaused = false;

    function selectRecordingMimeType() {{
      if (!recordingSupported || typeof MediaRecorder.isTypeSupported !== 'function') {{
        return '';
      }}
      return recordingMimeCandidates.find(type => MediaRecorder.isTypeSupported(type)) || '';
    }}

    function setRecordingState(state) {{
      recordingState = state;
      recorderControls.dataset.state = state;
      recorderControls.dataset.evaluationPaused = recordingGatePaused ? 'true' : 'false';
      recordStart.disabled = state !== 'idle' || !hasFrame || !recordingSupported ||
        recordingGatePaused;
      recordPause.disabled = recordingGatePaused || recordingAutomaticallyPaused ||
        (state !== 'recording' && state !== 'paused');
      recordStop.disabled = state !== 'recording' && state !== 'paused';
      recordPause.textContent = recordingGatePaused &&
        (state === 'recording' || state === 'paused') ?
        '评估中' : (state === 'paused' ? '▶ 继续' : 'Ⅱ 暂停');
    }}

    function reconcileRecordingGate() {{
      if (mediaRecorder === null || mediaRecorder.state === 'inactive') {{
        recordingAutomaticallyPaused = false;
        setRecordingState(recordingState);
        return;
      }}
      if (recordingGatePaused && mediaRecorder.state === 'recording') {{
        recordingAutomaticallyPaused = true;
        recordPause.disabled = true;
        mediaRecorder.pause();
        return;
      }}
      if (!recordingGatePaused && recordingAutomaticallyPaused) {{
        if (mediaRecorder.state === 'paused') {{
          recordingAutomaticallyPaused = false;
          recordPause.disabled = true;
          mediaRecorder.resume();
          return;
        }}
      }}
      setRecordingState(recordingState);
    }}

    function applyRecordingGate(paused) {{
      if (typeof paused !== 'boolean') {{
        throw new Error('recording gate must be a boolean');
      }}
      recordingGatePaused = paused;
      reconcileRecordingGate();
    }}
    window.visualizationApplyRecordingGate = applyRecordingGate;

    function markFrameReady() {{
      if (hasFrame) return;
      hasFrame = true;
      setRecordingState(recordingState);
    }}

    function releaseRecordingStream() {{
      if (recordingStream !== null) {{
        for (const track of recordingStream.getTracks()) track.stop();
      }}
      recordingStream = null;
    }}

    function saveRecording(chunks, mimeType) {{
      const effectiveType = mimeType || 'video/webm';
      const blob = new Blob(chunks, {{type: effectiveType}});
      if (blob.size === 0) {{
        console.error('recording produced no video data');
        return;
      }}
      const extension = effectiveType.toLowerCase().includes('mp4') ? 'mp4' : 'webm';
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `isaac-sim-recording-${{timestamp}}.${{extension}}`;
      anchor.style.display = 'none';
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 30_000);
    }}

    function startRecording() {{
      if (!recordingSupported || !hasFrame || recordingState !== 'idle' ||
          recordingGatePaused) return;
      try {{
        recordingStream = frame.captureStream();
        const mimeType = selectRecordingMimeType();
        mediaRecorder = mimeType ?
          new MediaRecorder(recordingStream, {{mimeType}}) :
          new MediaRecorder(recordingStream);
        const recorder = mediaRecorder;
        recordingChunks = [];
        recorder.addEventListener('dataavailable', event => {{
          if (event.data && event.data.size > 0) recordingChunks.push(event.data);
        }});
        recorder.addEventListener('pause', () => {{
          if (mediaRecorder === recorder) {{
            setRecordingState('paused');
            reconcileRecordingGate();
          }}
        }});
        recorder.addEventListener('resume', () => {{
          if (mediaRecorder === recorder) {{
            setRecordingState('recording');
            reconcileRecordingGate();
          }}
        }});
        recorder.addEventListener('error', event => {{
          console.error('browser recording failed', event.error || event);
        }});
        recorder.addEventListener('stop', () => {{
          const chunks = recordingChunks;
          const outputType = recorder.mimeType;
          releaseRecordingStream();
          if (mediaRecorder === recorder) mediaRecorder = null;
          recordingChunks = [];
          recordingAutomaticallyPaused = false;
          setRecordingState('idle');
          saveRecording(chunks, outputType);
        }});
        recorder.start(1000);
        setRecordingState('recording');
        reconcileRecordingGate();
      }} catch (error) {{
        console.error('could not start browser recording', error);
        releaseRecordingStream();
        mediaRecorder = null;
        recordingChunks = [];
        recordStart.title = '当前浏览器无法启动视频录制';
        setRecordingState('idle');
      }}
    }}

    function toggleRecordingPause() {{
      if (mediaRecorder === null || recordingGatePaused) return;
      recordingAutomaticallyPaused = false;
      if (mediaRecorder.state === 'recording') {{
        recordPause.disabled = true;
        mediaRecorder.pause();
      }} else if (mediaRecorder.state === 'paused') {{
        recordPause.disabled = true;
        mediaRecorder.resume();
      }}
    }}

    function stopRecording() {{
      if (mediaRecorder === null || mediaRecorder.state === 'inactive') return;
      setRecordingState('stopping');
      mediaRecorder.stop();
    }}

    recorderControls.addEventListener('pointerdown', event => event.stopPropagation());
    recordStart.addEventListener('click', startRecording);
    recordPause.addEventListener('click', toggleRecordingPause);
    recordStop.addEventListener('click', stopRecording);
    if (!recordingSupported) {{
      recorderControls.title = '当前浏览器不支持 canvas 视频录制';
    }}
    setRecordingState('idle');

    const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

    async function receiveFrames() {{
      while (true) {{
        let nextUrl = null;
        try {{
          const response = await fetch(`/frame.jpg?after=${{frameSequence}}`, {{
            cache: 'no-store'
          }});
          if (response.status === 204) continue;
          if (!response.ok) throw new Error(response.statusText);
          const nextSequence = Number(response.headers.get('X-Frame-Sequence'));
          const blob = await response.blob();
          nextUrl = URL.createObjectURL(blob);
          const probe = new Image();
          probe.src = nextUrl;
          await probe.decode();
          if (!hasFrame || recordingState === 'idle') {{
            if (frame.width !== probe.naturalWidth || frame.height !== probe.naturalHeight) {{
              frame.width = probe.naturalWidth;
              frame.height = probe.naturalHeight;
            }}
          }}
          latestFrameImage = probe;
          requestCompositedFrame();
          markFrameReady();
          if (Number.isSafeInteger(nextSequence) && nextSequence > frameSequence) {{
            frameSequence = nextSequence;
          }} else {{
            frameSequence += 1;
          }}
        }} catch (error) {{
          console.error('frame receive failed', error);
          await delay(250);
        }} finally {{
          if (nextUrl !== null) URL.revokeObjectURL(nextUrl);
        }}
      }}
    }}

    connectControls();
    receiveDashboardEvents();
    if (browserThreeScene === null) {{
      receivePresentationState();
      receiveFrames();
    }} else if (window.visualizationThreeAssetsReady === true) {{
      receivePresentationState();
    }} else {{
      window.addEventListener(
        'visualization-three-assets-ready',
        () => receivePresentationState(),
        {{once: true}}
      );
    }}
    requestCompositedFrame();
  </script>
{three_renderer_module}
</body>
</html>
""".encode(
        "utf-8"
    )


__all__ = ["build_browser_page"]
