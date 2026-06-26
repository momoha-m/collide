#include "ofApp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <sys/stat.h>
#include <unordered_map>

namespace {
constexpr std::uint32_t kCacheVersion = 7;
constexpr std::uint32_t kCacheDirectorySignatureVersion = 6;
constexpr std::uint32_t kFrameCacheMagic = 0x43565043; // CVPC
constexpr float kCameraInitialDistance = 0.85f;
constexpr float kCameraInitialMinDistance = 0.24f;
constexpr float kCameraInitialMaxDistance = 3.2f;
constexpr float kCameraFocusDistanceFactor = 1.35f;
constexpr float kCameraDollyDistance = 0.025f;
constexpr float kCameraMaxDistance = 80.0f;
constexpr float kCameraNearClip = 0.0001f;
constexpr float kCameraFarClip = 10000.0f;

struct VoxelAccumulator {
	glm::vec3 sum = glm::vec3(0.0f);
	int count = 0;
};

struct FlowVoxelAccumulator {
	glm::vec3 positionSum = glm::vec3(0.0f);
	glm::vec3 velocitySum = glm::vec3(0.0f);
	int count = 0;
};

using VoxelKey = std::array<int, 3>;

struct VoxelKeyHash {
	std::size_t operator()(const VoxelKey& key) const {
		std::size_t seed = 0;
		for(int value : key) {
			seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

std::uint64_t fnv1aAppend(std::uint64_t seed, const void* data, std::size_t size);

std::uint64_t fnv1aAppend(std::uint64_t seed, const void* data, std::size_t size) {
	const auto* bytes = static_cast<const std::uint8_t*>(data);
	for(std::size_t i = 0; i < size; ++i) {
		seed ^= static_cast<std::uint64_t>(bytes[i]);
		seed *= 1099511628211ull;
	}
	return seed;
}

std::uint64_t fnv1aString(std::uint64_t seed, const std::string& value) {
	return fnv1aAppend(seed, value.data(), value.size());
}

template <typename T>
std::uint64_t fnv1aValue(std::uint64_t seed, const T& value) {
	return fnv1aAppend(seed, &value, sizeof(T));
}

template <typename T>
bool readBinary(std::ifstream& input, T& value) {
	input.read(reinterpret_cast<char*>(&value), sizeof(T));
	return static_cast<bool>(input);
}

template <typename T>
bool writeBinary(std::ofstream& output, const T& value) {
	output.write(reinterpret_cast<const char*>(&value), sizeof(T));
	return static_cast<bool>(output);
}

bool fileLooksLocallyAvailable(const std::string& path) {
	struct stat fileStat {};
	if(::stat(path.c_str(), &fileStat) != 0) {
		return false;
	}
	if(!S_ISREG(fileStat.st_mode) || fileStat.st_size <= 0) {
		return false;
	}

	// Dropbox cloud-only placeholders report the logical size but no local blocks.
	return fileStat.st_blocks > 0;
}

bool shouldLogFrameNumber(int frameNumber) {
	return frameNumber < 10 || frameNumber % 100 == 0;
}

bool typeUsedForDisplayNormalization(int type) {
	return type == 0 || type == 2 || type == 3 || type == 4;
}

float packedTypeIntensityValue(int type, float intensity) {
	return static_cast<float>(type) + ofClamp(intensity, 0.0f, 1.0f) * 0.5f;
}

std::filesystem::path cacheOnlyPath() {
	if(const char* value = std::getenv("COLLIDE_VIS_CACHE_DIR")) {
		if(value[0] != '\0') {
			return value;
		}
	}
	return {};
}

std::vector<SnapshotFrameInfo> discoverCacheFrames(const std::filesystem::path& cachePath) {
	std::vector<SnapshotFrameInfo> frames;
	const std::regex pattern(R"(^frame-([0-9]+)\.bin$)");
	std::error_code ec;
	for(const auto& entry : std::filesystem::directory_iterator(cachePath, ec)) {
		if(ec || !entry.is_regular_file(ec)) {
			continue;
		}
		std::smatch match;
		const std::string filename = entry.path().filename().string();
		if(!std::regex_match(filename, match, pattern)) {
			continue;
		}
		SnapshotFrameInfo frame;
		frame.frameNumber = std::stoi(match[1].str());
		frame.filePath = entry.path().string();
		frames.push_back(std::move(frame));
	}
	std::sort(frames.begin(), frames.end(), [](const SnapshotFrameInfo& a, const SnapshotFrameInfo& b) {
		return a.frameNumber < b.frameNumber;
	});
	return frames;
}
}

void ofApp::setup() {
	ofSetWindowTitle("collide-vis-001");
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofBackground(0);
	ofEnableDepthTest();
	ofDisableArbTex();
	glEnable(GL_PROGRAM_POINT_SIZE);

	setupGui();
	setupPointShader();
	setupPtsm();

	resetCameraView();
	cameraTimeline.setup(&cam);
	cameraTimeline.load(cameraTimelinePath);
	loadRotationCues(rotationCuesPath);

	ofLogNotice() << "[collide-vis] Setup started";
	dataDirectoryPath = resolveDataDirectoryPath();
	ofLogNotice() << "[collide-vis] Dataset path: " << dataDirectoryPath;
	if(const auto cachePath = cacheOnlyPath(); !cachePath.empty()) {
		frameInfos = discoverCacheFrames(cachePath);
		ofLogNotice() << "[collide-vis] Cache-only mode: " << cachePath.string();
	} else {
		frameInfos = loader.discoverFrames(dataDirectoryPath);
	}
	if(frameInfos.empty()) {
		statusMessage = "No snapshot_*.hdf5 files found";
		ofLogError() << "[collide-vis] " << statusMessage << " in " << dataDirectoryPath;
		return;
	}

	ofLogNotice() << "[collide-vis] Discovered " << frameInfos.size() << " snapshot frames";
	ofLogNotice() << "[collide-vis] Using cache version " << kCacheVersion
		<< ", cache particle limit " << cacheParticleLimit;

	if(!preloadAllFrameData()) {
		statusMessage = "Failed to prepare render cache";
		ofLogError() << "[collide-vis] " << statusMessage;
		return;
	}

	preloadReady = true;
	resetCameraView();
	syncFrameState();
	lastUpdateTime = ofGetElapsedTimef();
	statusMessage = "Ready";
}

void ofApp::update() {
	const double now = ofGetElapsedTimef();
	const double deltaSeconds = std::max(0.0, now - lastUpdateTime);
	lastUpdateTime = now;

	if(!preloadReady || renderFrames.empty()) {
		return;
	}

	finishNextFramePrefetch();

	if(autoRotateParam) {
		rotationDegrees += static_cast<float>(deltaSeconds) * 6.0f;
	}

	const double previousPlaybackFramePosition = playbackFramePosition;
	if(playing) {
		updatePlaybackPosition(deltaSeconds);
	}
	applyPlaybackPosition(true);
	applyRotationCues(playbackFramePosition);
	cameraTimeline.apply(playbackFramePosition);
	constrainCameraDistance();
	syncDisplaySettings();
	prepareAheadFrameTexture();
	syncPtsmSettings();
	rebuildPtsmFieldIfNeeded();
	if(ptsmGui.enabled && ptsmEngine.getProbeCount() > 0) {
		ptsmEngine.setFramePosition(currentFrameBlend);
		if(playing) {
			const double frameCount = static_cast<double>(std::max<std::size_t>(1, renderFrames.size()));
			double playbackDeltaFrames = playbackFramePosition - previousPlaybackFramePosition;
			if(playbackDeltaFrames < -frameCount * 0.5) {
				playbackDeltaFrames += frameCount;
			} else if(playbackDeltaFrames > frameCount * 0.5) {
				playbackDeltaFrames -= frameCount;
			}
			const float referenceDeltaFrames = 3.0f / 60.0f;
			const float playbackScale = referenceDeltaFrames > 0.0f
				? static_cast<float>(std::abs(playbackDeltaFrames)) / referenceDeltaFrames
				: 1.0f;
			if(playbackScale > 0.0f) {
				const float ptsmUpdateDt =
					ptsmSettings.probe.dt *
					static_cast<float>(std::max(1, ptsmSettings.probe.substeps)) *
					playbackScale;
				ptsmEngine.update(ptsmUpdateDt);
				applyPtsmFlowForces(ptsmUpdateDt);
				updatePtsmDisplayState(deltaSeconds);
			}
		}
		updateCameraFromPtsmProbe();
		sendPtsmMetrics();
	}
}

void ofApp::draw() {
	ofBackground(0);

	if(preloadReady && frameTexture.isAllocated()) {
		cam.begin();
		ofPushMatrix();
		if(cameraMode == 0 && (autoRotateParam || std::abs(rotationDegrees) > 0.0001f)) {
			ofRotateYDeg(rotationDegrees);
		}
		if(showBoundsParam) {
			drawBounds();
		}
		drawPointCloud();
		drawPtsmProbes();
		ofPopMatrix();
		cam.end();
	}

	if(showGuiParam) {
		ofDisableDepthTest();
		gui.draw();
		ofEnableDepthTest();
	}
	drawHud();
}

void ofApp::exit() {
	waitForPendingPrefetch();
}

void ofApp::keyPressed(int key) {
	if(key == ' ') {
		playing = !playing;
		return;
	}
	if(key == OF_KEY_RIGHT && preloadReady && !renderFrames.empty()) {
		playbackFramePosition += 1.0;
		applyPlaybackPosition(true);
		return;
	}
	if(key == OF_KEY_LEFT && preloadReady && !renderFrames.empty()) {
		playbackFramePosition -= 1.0;
		applyPlaybackPosition(true);
		return;
	}
	if(key == 'g') {
		showGuiParam = !showGuiParam.get();
		return;
	}
	if(key == 'k') {
		cameraTimeline.addKeyframe(playbackFramePosition);
		return;
	}
	if(key == 'a') {
		addRotationCue();
		return;
	}
	if(key == 'v') {
		rotationCueEnabled = !rotationCueEnabled;
		statusMessage = rotationCueEnabled ? "Rotation cues on" : "Rotation cues off";
		return;
	}
	if(key == 'x') {
		removeNearestRotationCue();
		return;
	}
	if(key == '-') {
		adjustRotationDegrees(-5.0f);
		return;
	}
	if(key == '=') {
		adjustRotationDegrees(5.0f);
		return;
	}
	if(key == 'c') {
		cameraTimeline.toggleEnabled();
		return;
	}
	if(key == 'm') {
		cameraTimeline.toggleMode();
		return;
	}
	if(key == '[') {
		cameraTimeline.addCueInfluenceFrames(-30.0f);
		rotationCueInfluenceFrames = std::max(1.0f, rotationCueInfluenceFrames - 30.0f);
		return;
	}
	if(key == ']') {
		cameraTimeline.addCueInfluenceFrames(30.0f);
		rotationCueInfluenceFrames += 30.0f;
		return;
	}
	if(key == 's') {
		const bool cameraSaved = cameraTimeline.save(cameraTimelinePath);
		const bool rotationSaved = saveRotationCues(rotationCuesPath);
		if(cameraSaved && rotationSaved) {
			statusMessage = "Saved camera path and rotation cues";
		} else {
			statusMessage = "Failed to save camera path or rotation cues";
		}
		return;
	}
	if(key == 'l') {
		const bool cameraLoaded = cameraTimeline.load(cameraTimelinePath);
		const bool rotationLoaded = loadRotationCues(rotationCuesPath);
		if(cameraLoaded || rotationLoaded) {
			statusMessage = "Loaded camera path / rotation cues";
		} else {
			statusMessage = "Failed to load camera path / rotation cues";
		}
		return;
	}
	if(key == 'd') {
		cameraTimeline.removeNearestKeyframe(playbackFramePosition);
		return;
	}
	if(key == 'r') {
		resetCameraView();
		cameraTimeline.refreshBaseCamera();
		return;
	}
	if(key == 'R') {
		resetAllState();
		return;
	}
	if(key == 'f') {
		fullResolutionParam = !fullResolutionParam.get();
		return;
	}
	if(key == 'p') {
		performanceModeParam = !performanceModeParam.get();
		return;
	}
	if(key == 'i') {
		spawnPtsmProbe();
		return;
	}
	if(key == 'N') {
		ptsmCurrentSpawnPosition.reset();
		ptsmCurrentSpawnVelocity.reset();
		ptsmLockedSpawnPosition.reset();
		ptsmLockedSpawnVelocity.reset();
		spawnPtsmProbe();
		return;
	}
	if(key == 'B') {
		lockPtsmSpawn();
		return;
	}
	if(key == 'I') {
		clearPtsmProbes();
		return;
	}
	if(key == 'V') {
		cameraMode = (cameraMode + 1) % 2;
		cameraStateInitialized = false;
		if(cameraMode == 0) {
			resetCameraView();
			cameraTimeline.refreshBaseCamera();
			cam.enableMouseInput();
		} else {
			cam.disableMouseInput();
			updateCameraFromPtsmProbe();
		}
		statusMessage = "Camera " + cameraModeLabel();
		return;
	}
	if(key == '0') {
		showGasParam = !showGasParam.get();
		return;
	}
	if(key == '1') {
		showDarkMatterParam = !showDarkMatterParam.get();
		return;
	}
	if(key == '2') {
		showDiskParam = !showDiskParam.get();
		return;
	}
	if(key == '3') {
		showBulgeParam = !showBulgeParam.get();
		return;
	}
	if(key == '4') {
		showStarsParam = !showStarsParam.get();
		return;
	}
}

void ofApp::windowResized(int w, int h) {
	(void)w;
	(void)h;
	gui.setPosition(18, 18);
}

void ofApp::setupGui() {
	guiParams.setName("collide-vis");
	guiParams.add(pointSizeParam.set("pointSize", 1.0f, 0.6f, 8.0f));
	guiParams.add(sceneScaleParam.set("sceneScale", 1.0f, 0.3f, 8.0f));
	guiParams.add(coreScaleParam.set("viewScale", 1.0f, 0.2f, 8.0f));
	guiParams.add(playbackFpsParam.set("playbackFps", 3.0f, 0.25f, 30.0f));
	guiParams.add(temporalSmoothParam.set("smoothMotion", true));
	guiParams.add(brightnessParam.set("brightness", 1.45f, 0.1f, 6.0f));
	guiParams.add(gradientParam.set("gradient", 0.32f, 0.0f, 1.0f));
	guiParams.add(typeColorParam.set("typeColor", 0.65f, 0.0f, 1.0f));
	guiParams.add(maxDrawCountParam.set("drawCap", 300000, 1000, 2000000));
	guiParams.add(fullResolutionParam.set("fullRes", true));
	guiParams.add(autoLodParam.set("autoLOD", true));
	guiParams.add(densityCompensationParam.set("densityBoost", true));
	guiParams.add(depthParticlesParam.set("depthParticles", true));
	guiParams.add(performanceModeParam.set("performance", true));
	guiParams.add(showGasParam.set("gas", true));
	guiParams.add(showDarkMatterParam.set("darkMatter", false));
	guiParams.add(showDiskParam.set("disk", true));
	guiParams.add(showBulgeParam.set("bulge", true));
	guiParams.add(showStarsParam.set("stars", true));
	guiParams.add(showBoundsParam.set("bounds", false));
	guiParams.add(autoRotateParam.set("autoRotate", false));
	guiParams.add(showGuiParam.set("gui", true));
	ptsmSettings = ptsm::Settings::paper3D();
	ptsmSettings.field.sigma = 0.62f;
	ptsmSettings.field.fieldGain = 45.0f;
	ptsmSettings.field.sampleCount = 4000;
	ptsmSettings.field.normalizeByAttractorCount = true;
	ptsmSettings.probe.dt = 0.0005f;
	ptsmSettings.probe.energyRetention = 0.99945f;
	ptsmSettings.probe.substeps = 4;
	ptsmSettings.probe.trailLength = 12000;
	ptsmSettings.probe.boundsEnabled = true;
	ptsmSettings.probe.boundsMin = glm::vec3(-4.0f);
	ptsmSettings.probe.boundsMax = glm::vec3(4.0f);
	ptsmSettings.probe.boundaryBounce = 0.92f;
	ptsmSettings.probe.maxForce = 80.0f;
	ptsmSettings.probe.maxSpeed = 4.2f;
	ptsmSettings.injection.maxKineticEnergy = 1.35f;
	ptsmSettings.injection.energyRatio = 0.94f;
	ptsmSettings.osc.enabled = false;
	ptsmGui.setup(ptsmSettings, "ptsm");
	guiParams.add(ptsmGui.parameters);
	guiParams.add(ptsmDensityMixParam.set("ptsm density mix", 0.18f, 0.0f, 1.0f));
	guiParams.add(ptsmFlowCouplingParam.set("ptsm flow follow", 16.0f, 0.0f, 80.0f));
	guiParams.add(ptsmFlowRadiusParam.set("ptsm flow radius", 0.38f, 0.05f, 3.0f));
	guiParams.add(ptsmFlowVelocityScaleParam.set("ptsm flow scale", 130.0f, 0.0f, 300.0f));
	guiParams.add(ptsmFlowVerticalMixParam.set("ptsm flow vertical", 0.68f, 0.0f, 1.0f));
	guiParams.add(ptsmTidalGainParam.set("ptsm tidal gain", 22.0f, 0.0f, 120.0f));
	guiParams.add(ptsmVorticityGainParam.set("ptsm vortex gain", 18.0f, 0.0f, 120.0f));
	guiParams.add(ptsmCompressionGainParam.set("ptsm compress gain", 16.0f, 0.0f, 120.0f));
	guiParams.add(ptsmDispersionGainParam.set("ptsm disperse gain", 9.0f, 0.0f, 120.0f));
	guiParams.add(ptsmProbeRadiusParam.set("ptsm radius", ptsmProbeRadius, 0.001f, 0.04f));
	guiParams.add(ptsmTrailAlphaParam.set("ptsm trail alpha", ptsmTrailAlpha, 0.0f, 255.0f));
	guiParams.add(ptsmTrailSmoothingParam.set("ptsm trail smooth", ptsmTrailSmoothingSize, 0, 20));
	gui.setup(guiParams);
	gui.setPosition(18, 18);

	previousMaxDrawCount = maxDrawCountParam;
	previousFullResolution = fullResolutionParam;
	previousAutoLod = autoLodParam;
}

void ofApp::setupPointShader() {
	const std::string vertexShader = R"(
		#version 150

		uniform mat4 modelViewMatrix;
		uniform mat4 projectionMatrix;
		uniform sampler2D positionTexture;
		uniform sampler2D nextPositionTexture;
		uniform int textureWidth;
		uniform int displayStride;
		uniform float pointSize;
		uniform float pointDensityScale;
		uniform float sceneScale;
		uniform float viewScale;
		uniform vec3 displayNormalizationCenter;
		uniform float displayNormalizationScale;
		uniform float frameBlend;
		uniform float gradientMix;
		uniform float typeColorMix;
		uniform int showGas;
		uniform int showDarkMatter;
		uniform int showDisk;
		uniform int showBulge;
		uniform int showStars;
		uniform int performanceMode;
		uniform int temporalSmooth;

		in vec4 position;
		out float vType;
		out float vIntensity;
		out float vMotion;
		out float vDepthCue;
		out float vVisible;
		out vec3 vFastColor;
		out float vFastAlpha;

		bool typeEnabled(float typeValue) {
			if(typeValue < 0.5) return showGas != 0;
			if(typeValue < 1.5) return showDarkMatter != 0;
			if(typeValue < 2.5) return showDisk != 0;
			if(typeValue < 3.5) return showBulge != 0;
			if(typeValue < 4.5) return showStars != 0;
			return false;
		}

		vec3 fastBlueGradient(float typeValue, float intensity, float motion) {
			vec3 outerBlue = vec3(0.10, 0.30, 0.78);
			vec3 referenceBlue = vec3(0.22, 0.56, 1.0);
			vec3 coreBlue = vec3(0.62, 0.86, 1.0);
			float signal = clamp(intensity * 0.65 + motion * 0.28, 0.0, 1.0);
			vec3 color = mix(referenceBlue, coreBlue, signal);
			if(typeValue < 1.5 && typeValue >= 0.5) {
				color = mix(outerBlue, referenceBlue, 0.42);
			}
			return color;
		}

		vec3 fastPotentialGradient(float signal) {
			vec3 cyan = vec3(0.10, 0.95, 1.0);
			vec3 warm = vec3(1.0, 0.49, 0.10);
			return mix(cyan, warm, clamp(signal, 0.0, 1.0));
		}

		vec3 fastTypeColor(float typeValue, float intensity) {
			vec3 gasCold = vec3(0.08, 0.72, 1.00);
			vec3 gasHot = vec3(1.00, 0.43, 0.10);
			vec3 darkMatter = vec3(0.20, 0.26, 0.74);
			vec3 disk = vec3(1.00, 0.70, 0.18);
			vec3 bulge = vec3(1.00, 0.38, 0.50);
			vec3 stars = vec3(1.00, 0.93, 0.68);
			if(typeValue < 0.5) return mix(gasCold, gasHot, clamp(intensity, 0.0, 1.0));
			if(typeValue < 1.5) return darkMatter;
			if(typeValue < 2.5) return disk;
			if(typeValue < 3.5) return bulge;
			if(typeValue < 4.5) return stars;
			return vec3(0.86, 0.92, 1.00);
		}

		float typeAlpha(float typeValue) {
			if(typeValue < 0.5) return 1.00;
			if(typeValue < 1.5) return 0.24;
			if(typeValue < 2.5) return 0.96;
			if(typeValue < 3.5) return 0.88;
			if(typeValue < 4.5) return 1.00;
			return 0.90;
		}

		void main() {
			int particleIndex = gl_VertexID * displayStride;
			ivec2 texel = ivec2(particleIndex % textureWidth, particleIndex / textureWidth);
			vec4 current = texelFetch(positionTexture, texel, 0);
			vec4 next = current;
			float motionBlend = temporalSmooth != 0 ? frameBlend : 0.0;
			if(motionBlend > 0.0001) {
				next = texelFetch(nextPositionTexture, texel, 0);
			}
			float typeValue = floor(current.a + 0.0001);
			vIntensity = clamp((current.a - typeValue) * 2.0, 0.0, 1.0);
			vVisible = typeEnabled(typeValue) ? 1.0 : 0.0;
			vec3 currentWorld = (current.rgb - displayNormalizationCenter) * displayNormalizationScale;
			vec3 nextWorld = (next.rgb - displayNormalizationCenter) * displayNormalizationScale;
			vec3 world = mix(currentWorld, nextWorld, motionBlend);
			world *= sceneScale * viewScale;
			vec4 view = modelViewMatrix * vec4(world, 1.0);
			float typeSize = typeValue < 0.5 ? 1.20 : (typeValue < 1.5 ? 0.88 : 1.05);
			float perspectiveScale = 1.0;
			float motion = 0.0;
			vDepthCue = 1.0;
			if(performanceMode == 0) {
				float viewDepth = max(-view.z, 0.06);
				perspectiveScale = clamp(0.92 / sqrt(viewDepth), 0.42, 2.35);
				motion = clamp(length(nextWorld - currentWorld) * 18.0, 0.0, 1.0);
				vDepthCue = clamp(1.28 / sqrt(viewDepth), 0.34, 1.35);
			}
			vType = typeValue;
			vMotion = motion;
			vec3 fastBase = fastBlueGradient(typeValue, vIntensity, motion);
			vec3 fastGradient = fastPotentialGradient(vIntensity * 0.45 + motion * 0.20 + 0.22);
			vec3 fastTyped = fastTypeColor(typeValue, vIntensity);
			vFastColor = mix(
				mix(fastBase, fastGradient, clamp(gradientMix, 0.0, 1.0)),
				fastTyped,
				clamp(typeColorMix, 0.0, 1.0)
			);
			vFastAlpha = typeAlpha(typeValue);
			gl_Position = projectionMatrix * view;
			float intensitySize = performanceMode == 0
				? 1.0 + vIntensity * 0.42 + motion * 0.10
				: 1.0 + vIntensity * 0.18;
			gl_PointSize = vVisible > 0.5
				? max(
					pointSize * pointDensityScale * typeSize * perspectiveScale * intensitySize,
					performanceMode == 0 ? 1.1 : 1.6
				)
				: 0.0;
		}
	)";

	const std::string fragmentShader = R"(
		#version 150

		uniform float brightness;
		uniform float densityGain;
		uniform float gradientMix;
		uniform float typeColorMix;
		uniform int fastFullRes;
		in float vType;
		in float vIntensity;
		in float vMotion;
		in float vDepthCue;
		in float vVisible;
		in vec3 vFastColor;
		in float vFastAlpha;
		out vec4 fragColor;

		vec3 blueGradient(float typeValue, float intensity, float motion) {
			vec3 outerBlue = vec3(0.10, 0.30, 0.78);
			vec3 referenceBlue = vec3(0.22, 0.56, 1.0);
			vec3 coreBlue = vec3(0.62, 0.86, 1.0);
			float signal = clamp(intensity * 0.65 + motion * 0.28, 0.0, 1.0);
			vec3 color = mix(referenceBlue, coreBlue, signal);
			if(typeValue < 1.5 && typeValue >= 0.5) {
				color = mix(outerBlue, referenceBlue, 0.42);
			}
			return color;
		}

		vec3 potentialGradient(float signal) {
			vec3 cyan = vec3(0.10, 0.95, 1.0);
			vec3 warm = vec3(1.0, 0.49, 0.10);
			return mix(cyan, warm, smoothstep(0.05, 1.0, signal));
		}

		vec3 fastPotentialGradient(float signal) {
			vec3 cyan = vec3(0.10, 0.95, 1.0);
			vec3 warm = vec3(1.0, 0.49, 0.10);
			return mix(cyan, warm, clamp(signal, 0.0, 1.0));
		}

		vec3 typeColor(float typeValue, float intensity, float motion) {
			vec3 gasCold = vec3(0.08, 0.72, 1.00);
			vec3 gasHot = vec3(1.00, 0.43, 0.10);
			vec3 darkMatter = vec3(0.20, 0.26, 0.74);
			vec3 disk = vec3(1.00, 0.70, 0.18);
			vec3 bulge = vec3(1.00, 0.38, 0.50);
			vec3 stars = vec3(1.00, 0.93, 0.68);
			if(typeValue < 0.5) {
				return mix(gasCold, gasHot, smoothstep(0.06, 0.92, max(intensity, motion * 0.55)));
			}
			if(typeValue < 1.5) return darkMatter;
			if(typeValue < 2.5) return disk;
			if(typeValue < 3.5) return bulge;
			if(typeValue < 4.5) return stars;
			return vec3(0.86, 0.92, 1.00);
		}

		vec3 fastTypeColor(float typeValue, float intensity) {
			vec3 gasCold = vec3(0.08, 0.72, 1.00);
			vec3 gasHot = vec3(1.00, 0.43, 0.10);
			vec3 darkMatter = vec3(0.20, 0.26, 0.74);
			vec3 disk = vec3(1.00, 0.70, 0.18);
			vec3 bulge = vec3(1.00, 0.38, 0.50);
			vec3 stars = vec3(1.00, 0.93, 0.68);
			if(typeValue < 0.5) return mix(gasCold, gasHot, clamp(intensity, 0.0, 1.0));
			if(typeValue < 1.5) return darkMatter;
			if(typeValue < 2.5) return disk;
			if(typeValue < 3.5) return bulge;
			if(typeValue < 4.5) return stars;
			return vec3(0.86, 0.92, 1.00);
		}

		float typeAlpha(float typeValue) {
			if(typeValue < 0.5) return 1.00;
			if(typeValue < 1.5) return 0.24;
			if(typeValue < 2.5) return 0.96;
			if(typeValue < 3.5) return 0.88;
			if(typeValue < 4.5) return 1.00;
			return 0.90;
		}

		void main() {
			if(vVisible < 0.5) {
				discard;
			}
			vec2 centered = gl_PointCoord * 2.0 - 1.0;
			float radius2 = dot(centered, centered);
			float glow = exp(-radius2 * 2.6);
			float gain = max(densityGain, 1.0);
			float brightnessGain = sqrt(gain);
			float alphaGain = sqrt(gain);
			if(fastFullRes != 0) {
				fragColor = vec4(
					vFastColor * glow * brightness * brightnessGain * vDepthCue,
					clamp(glow * 0.42 * vFastAlpha * alphaGain, 0.0, 1.0)
				);
				return;
			}
			float gradientSignal = clamp(
				vIntensity * 0.42 +
				vMotion * 0.22 +
				smoothstep(0.34, 1.0, glow) * 0.42,
				0.0,
				1.0
			);
			vec3 baseColor = blueGradient(vType, vIntensity, vMotion);
			vec3 gradColor = potentialGradient(gradientSignal);
			vec3 mixedGradient = mix(baseColor, gradColor, clamp(gradientMix, 0.0, 1.0));
			vec3 typedColor = typeColor(vType, vIntensity, vMotion);
			vec3 color = mix(mixedGradient, typedColor, clamp(typeColorMix, 0.0, 1.0)) * glow;
			float alpha = glow * 0.42 * typeAlpha(vType);
			fragColor = vec4(
				color * brightness * brightnessGain * vDepthCue,
				clamp(alpha * alphaGain, 0.0, 1.0)
			);
		}
	)";

	pointShader.setupShaderFromSource(GL_VERTEX_SHADER, vertexShader);
	pointShader.setupShaderFromSource(GL_FRAGMENT_SHADER, fragmentShader);
	pointShader.bindDefaults();
	pointShader.linkProgram();
}

void ofApp::setupPtsm() {
	ptsmEngine.setFieldModel(&ptsmField);
	ptsmProbeSphere.setRadius(ptsmProbeRadius);
	ptsmProbeSphere.setResolution(8);
	syncPtsmSettings();
}

std::string ofApp::resolveDataDirectoryPath() const {
	if(const char* envPath = std::getenv("COLLIDE_VIS_DATA")) {
		const std::string value(envPath);
		if(!value.empty()) {
			return value;
		}
	}

	const std::filesystem::path configPath(ofToDataPath("dataset_path.txt", true));
	std::ifstream config(configPath);
	if(config.is_open()) {
		std::string line;
		std::getline(config, line);
		if(!line.empty()) {
			return line;
		}
	}

	return defaultDataDirectoryPath;
}

bool ofApp::preloadAllFrameData() {
	if(frameInfos.empty()) {
		return false;
	}

	const auto preloadStart = std::chrono::steady_clock::now();
	cacheDirectory = buildCacheDirectory();
	ofLogNotice() << "[collide-vis] Render cache directory: " << cacheDirectory.string();
	ofLogNotice() << "[collide-vis] Preloading dataset before playback; HDF5 will not be used during playback";

	if(loadPreprocessedCache(cacheDirectory)) {
		const float preloadMs = std::chrono::duration<float, std::milli>(
			std::chrono::steady_clock::now() - preloadStart
		).count();
		ofLogNotice() << "[collide-vis] Preload finished from cache in " << ofToString(preloadMs, 1) << " ms";
		return true;
	}
	if(!cacheOnlyPath().empty()) {
		ofLogError() << "[collide-vis] Cache-only mode requires a complete valid cache at " << cacheDirectory;
		return false;
	}

	ofLogNotice() << "[collide-vis] No usable cache found. Building cache from local HDF5 files";
	std::error_code ec;
	std::filesystem::create_directories(cacheDirectory, ec);
	if(ec) {
		ofLogError() << "[collide-vis] Failed to create cache directory "
			<< cacheDirectory.string() << ": " << ec.message();
		return false;
	}

	std::size_t skippedOffline = 0;
	std::size_t skippedFailed = 0;
	std::vector<std::uint64_t> slotParticleIds;
	std::unordered_map<std::uint64_t, std::size_t> slotById;
	const std::size_t progressStep = std::max<std::size_t>(1, frameInfos.size() / 20);
	const glm::vec3 rawCenter(0.0f);
	const float rawScale = 1.0f;

	renderFrames.clear();
	renderFrames.reserve(frameInfos.size());

	for(std::size_t frameIndex = 0; frameIndex < frameInfos.size(); ++frameIndex) {
		const auto& info = frameInfos[frameIndex];
		if(!fileLooksLocallyAvailable(info.filePath)) {
			++skippedOffline;
			continue;
		}

		try {
			const SnapshotFrame frame = loader.loadFrame(info);
			if(frame.particles.empty() || !frame.boundsValid) {
				++skippedFailed;
				continue;
			}

			if(slotParticleIds.empty()) {
				slotParticleIds.reserve(frame.particles.size());
				for(const auto& particle : frame.particles) {
					slotParticleIds.push_back(particle.id);
				}
				std::sort(slotParticleIds.begin(), slotParticleIds.end());
				slotParticleIds.erase(
					std::unique(slotParticleIds.begin(), slotParticleIds.end()),
					slotParticleIds.end()
				);
				if(slotParticleIds.size() > cacheParticleLimit) {
					slotParticleIds.resize(cacheParticleLimit);
				}
				maxParticleCount = slotParticleIds.size();
				textureWidth = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(maxParticleCount))));
				textureHeight = static_cast<int>(
					(maxParticleCount + static_cast<std::size_t>(textureWidth) - 1) /
					static_cast<std::size_t>(textureWidth)
				);
				slotById.reserve(slotParticleIds.size() * 2);
				for(std::size_t slot = 0; slot < slotParticleIds.size(); ++slot) {
					slotById[slotParticleIds[slot]] = slot;
				}
				ofLogNotice() << "[collide-vis] Fixed ParticleID slots: "
					<< slotParticleIds.size() << " from frame " << info.frameNumber;
				ofLogNotice() << "[collide-vis] Cache uses raw coordinates with display normalization; texture "
					<< textureWidth << "x" << textureHeight;
			}

			std::vector<RenderParticle> particles = buildRenderParticles(
				frame,
				rawCenter,
				rawScale,
				slotById,
				maxParticleCount
			);
			if(particles.empty()) {
				++skippedFailed;
				continue;
			}

			CachedFrameInfo cachedInfo;
			cachedInfo.sourceIndex = frameIndex;
			cachedInfo.frameNumber = frame.frameNumber;
			cachedInfo.time = frame.time;
			cachedInfo.particleCount = particles.size();
			for(const auto& particle : particles) {
				const int type = static_cast<int>(std::floor(particle.typeIntensity + 0.0001f));
				if(type >= 0 && type < static_cast<int>(cachedInfo.typeCounts.size())) {
					++cachedInfo.typeCounts[static_cast<std::size_t>(type)];
				}
			}

			renderFrames.push_back(cachedInfo);
			if(!writeCachedFrameData(renderFrames.size() - 1, particles)) {
				ofLogError() << "[collide-vis] Failed to write cache for frame " << frame.frameNumber;
				renderFrames.pop_back();
				return false;
			}
		} catch(const std::exception& e) {
			++skippedFailed;
			ofLogWarning() << "[collide-vis] Preload cache skip frame " << info.frameNumber
				<< ": " << e.what();
		}

		if((frameIndex + 1) % progressStep == 0 || frameIndex + 1 == frameInfos.size()) {
			ofLogNotice() << "[collide-vis] Preload fixed-ID cache phase: "
				<< (frameIndex + 1) << "/" << frameInfos.size()
				<< " cached=" << renderFrames.size()
				<< " offline=" << skippedOffline
				<< " failed=" << skippedFailed;
		}
	}

	if(renderFrames.empty()) {
		ofLogError() << "[collide-vis] Cache build produced no frames";
		return false;
	}

	if(!writePreprocessedCache(cacheDirectory)) {
		return false;
	}

	if(!computeDisplayNormalizationFromCache()) {
		ofLogWarning() << "[collide-vis] Failed to compute display normalization; using cached coordinates as-is";
	}
	warmInitialRamCache(0);
	const bool loaded = loadRenderFrame(0);
	const float preloadMs = std::chrono::duration<float, std::milli>(
		std::chrono::steady_clock::now() - preloadStart
	).count();
	ofLogNotice() << "[collide-vis] Preload finished in " << ofToString(preloadMs, 1)
		<< " ms; cached frames=" << renderFrames.size()
		<< ", skipped offline=" << skippedOffline
		<< ", skipped failed=" << skippedFailed;
	return loaded;
}

std::filesystem::path ofApp::buildCacheDirectory() const {
	if(const auto cachePath = cacheOnlyPath(); !cachePath.empty()) {
		return cachePath;
	}

	std::uint64_t signature = 14695981039346656037ull;
	signature = fnv1aString(signature, dataDirectoryPath);
	signature = fnv1aValue(signature, kCacheDirectorySignatureVersion);
	signature = fnv1aValue(signature, cacheParticleLimit);
	signature = fnv1aValue(signature, frameInfos.size());

	for(const auto& frame : frameInfos) {
		signature = fnv1aValue(signature, frame.frameNumber);
		signature = fnv1aString(signature, frame.filePath);
		std::error_code ec;
		const std::filesystem::path framePath(frame.filePath);
		if(std::filesystem::exists(framePath, ec)) {
			const auto size = std::filesystem::file_size(framePath, ec);
			if(!ec) {
				signature = fnv1aValue(signature, static_cast<std::uint64_t>(size));
			}
			const auto writeTime = std::filesystem::last_write_time(framePath, ec);
			if(!ec) {
				signature = fnv1aValue(signature, writeTime.time_since_epoch().count());
			}
		}
	}

	std::ostringstream name;
	name << "dataset-" << std::hex << std::setw(16) << std::setfill('0') << signature
		<< "-limit-" << std::dec << cacheParticleLimit;

	if(const char* envPath = std::getenv("COLLIDE_VIS_CACHE")) {
		const std::string value(envPath);
		if(!value.empty()) {
			return std::filesystem::path(value) / name.str();
		}
	}

	const std::string biwinPrefix = "/Volumes/BIWIN/";
	if(dataDirectoryPath.rfind(biwinPrefix, 0) == 0) {
		return std::filesystem::path("/Volumes/BIWIN/collide-vis-001-cache") / name.str();
	}

	const std::filesystem::path localCacheRoot(ofToDataPath("collide-vis-001-cache", true));
	const std::filesystem::path computedCacheDir = localCacheRoot / name.str();
	std::error_code ec;
	if(std::filesystem::exists(computedCacheDir / "manifest.txt", ec)) {
		return computedCacheDir;
	}

	const std::string limitSuffix = "-limit-" + ofToString(cacheParticleLimit);
	std::filesystem::path fallbackCacheDir;
	std::size_t fallbackCount = 0;
	if(std::filesystem::exists(localCacheRoot, ec)) {
		for(const auto& entry : std::filesystem::directory_iterator(localCacheRoot, ec)) {
			if(ec) {
				break;
			}
			if(!entry.is_directory(ec)) {
				continue;
			}
			const std::string dirname = entry.path().filename().string();
			if(dirname.rfind("dataset-", 0) != 0 ||
			   dirname.size() < limitSuffix.size() ||
			   dirname.compare(dirname.size() - limitSuffix.size(), limitSuffix.size(), limitSuffix) != 0) {
				continue;
			}

			std::ifstream manifest(entry.path() / "manifest.txt");
			std::uint32_t version = 0;
			std::size_t sourceFrameCount = 0;
			std::size_t cachedFrameCount = 0;
			int manifestTextureWidth = 0;
			int manifestTextureHeight = 0;
			glm::vec3 manifestMinBounds(0.0f);
			glm::vec3 manifestMaxBounds(0.0f);
			std::size_t manifestMaxParticleCount = 0;
			std::size_t manifestCacheParticleLimit = 0;
			manifest >> version;
			manifest >> sourceFrameCount;
			manifest >> cachedFrameCount;
			manifest >> manifestTextureWidth >> manifestTextureHeight;
			manifest >> manifestMinBounds.x >> manifestMinBounds.y >> manifestMinBounds.z;
			manifest >> manifestMaxBounds.x >> manifestMaxBounds.y >> manifestMaxBounds.z;
			manifest >> manifestMaxParticleCount;
			manifest >> manifestCacheParticleLimit;
			if(manifest &&
			   version == kCacheVersion &&
			   cachedFrameCount > 0 &&
			   manifestCacheParticleLimit == cacheParticleLimit) {
				fallbackCacheDir = entry.path();
				++fallbackCount;
			}
		}
	}

	if(fallbackCount == 1) {
		ofLogNotice() << "[collide-vis] Using existing local render cache "
			<< fallbackCacheDir.string()
			<< " because no cache exists for the current dataset path signature";
		return fallbackCacheDir;
	}

	return computedCacheDir;
}

bool ofApp::loadPreprocessedCache(const std::filesystem::path& cacheDir) {
	const std::filesystem::path manifestPath = cacheDir / "manifest.txt";
	if(!std::filesystem::exists(manifestPath)) {
		return false;
	}

	std::ifstream manifest(manifestPath);
	if(!manifest.is_open()) {
		return false;
	}

	std::uint32_t version = 0;
	std::size_t sourceFrameCount = 0;
	std::size_t cachedFrameCount = 0;
	std::size_t manifestMaxParticleCount = 0;
	std::size_t manifestCacheParticleLimit = 0;
	manifest >> version;
	manifest >> sourceFrameCount;
	manifest >> cachedFrameCount;
	manifest >> textureWidth >> textureHeight;
	manifest >> worldMin.x >> worldMin.y >> worldMin.z;
	manifest >> worldMax.x >> worldMax.y >> worldMax.z;
	manifest >> manifestMaxParticleCount;
	manifest >> manifestCacheParticleLimit;
	if(!manifest ||
	   version != kCacheVersion ||
	   manifestCacheParticleLimit != cacheParticleLimit ||
	   cachedFrameCount == 0) {
		return false;
	}
	if(sourceFrameCount != frameInfos.size()) {
		ofLogWarning() << "[collide-vis] Cache source frame count " << sourceFrameCount
			<< " differs from current local snapshot count " << frameInfos.size()
			<< "; using cached frames without rebuilding from HDF5";
	}

	maxParticleCount = manifestMaxParticleCount;
	renderFrames.clear();
	renderFrames.reserve(cachedFrameCount);
	for(std::size_t i = 0; i < cachedFrameCount; ++i) {
		CachedFrameInfo info;
		manifest >> info.sourceIndex;
		manifest >> info.frameNumber;
		manifest >> info.time;
		manifest >> info.particleCount;
		for(auto& typeCount : info.typeCounts) {
			manifest >> typeCount;
		}
		if(!manifest || info.particleCount == 0) {
			return false;
		}
		renderFrames.push_back(info);
	}

	const std::size_t progressStep = std::max<std::size_t>(1, renderFrames.size() / 10);
	for(std::size_t frameIndex = 0; frameIndex < renderFrames.size(); ++frameIndex) {
		if(!loadCachedFrameData(frameIndex, nullptr)) {
			ofLogWarning() << "[collide-vis] Cache manifest exists, but frame cache is missing/corrupt at index "
				<< frameIndex;
			renderFrames.clear();
			return false;
		}
		if((frameIndex + 1) % progressStep == 0 || frameIndex + 1 == renderFrames.size()) {
			ofLogNotice() << "[collide-vis] Cache verify phase: "
				<< (frameIndex + 1) << "/" << renderFrames.size() << " frames";
		}
	}

	ofLogNotice() << "[collide-vis] Loaded render cache manifest from " << cacheDir.string()
		<< " with " << renderFrames.size() << " cached frames";
	if(!computeDisplayNormalizationFromCache()) {
		ofLogWarning() << "[collide-vis] Failed to compute display normalization; using cached coordinates as-is";
	}
	warmInitialRamCache(0);
	return loadRenderFrame(0);
}

bool ofApp::writePreprocessedCache(const std::filesystem::path& cacheDir) const {
	const std::filesystem::path manifestPath = cacheDir / "manifest.txt";
	const std::filesystem::path tempPath = cacheDir / "manifest.tmp";
	std::ofstream manifest(tempPath, std::ios::trunc);
	if(!manifest.is_open()) {
		ofLogWarning() << "[collide-vis] Failed to write cache manifest: " << tempPath.string();
		return false;
	}

	manifest << kCacheVersion << '\n';
	manifest << frameInfos.size() << '\n';
	manifest << renderFrames.size() << '\n';
	manifest << textureWidth << ' ' << textureHeight << '\n';
	manifest << worldMin.x << ' ' << worldMin.y << ' ' << worldMin.z << '\n';
	manifest << worldMax.x << ' ' << worldMax.y << ' ' << worldMax.z << '\n';
	manifest << maxParticleCount << '\n';
	manifest << cacheParticleLimit << '\n';
	for(const auto& frame : renderFrames) {
		manifest << frame.sourceIndex << ' '
			<< frame.frameNumber << ' '
			<< std::setprecision(17) << frame.time << ' '
			<< frame.particleCount;
		for(std::size_t typeCount : frame.typeCounts) {
			manifest << ' ' << typeCount;
		}
		manifest << '\n';
	}
	manifest.close();
	if(!manifest) {
		ofLogWarning() << "[collide-vis] Failed to finalize cache manifest: " << tempPath.string();
		return false;
	}

	std::error_code ec;
	std::filesystem::rename(tempPath, manifestPath, ec);
	if(ec) {
		std::filesystem::remove(manifestPath, ec);
		ec.clear();
		std::filesystem::rename(tempPath, manifestPath, ec);
	}
	if(ec) {
		ofLogWarning() << "[collide-vis] Failed to move cache manifest into place: " << ec.message();
		return false;
	}

	ofLogNotice() << "[collide-vis] Wrote render cache manifest to " << manifestPath.string();
	return true;
}

bool ofApp::loadCachedFrameData(std::size_t frameIndex, std::vector<RenderParticle>* particles) const {
	if(frameIndex >= renderFrames.size()) {
		return false;
	}

	const auto& info = renderFrames[frameIndex];
	const std::filesystem::path framePath =
		cacheDirectory / ("frame-" + ofToString(info.frameNumber) + ".bin");
	std::ifstream input(framePath, std::ios::binary);
	if(!input.is_open()) {
		return false;
	}

	std::uint32_t magic = 0;
	std::uint32_t version = 0;
	int frameNumber = -1;
	double time = 0.0;
	std::uint32_t particleCount = 0;
	if(!readBinary(input, magic) ||
	   !readBinary(input, version) ||
	   !readBinary(input, frameNumber) ||
	   !readBinary(input, time) ||
	   !readBinary(input, particleCount)) {
		return false;
	}

	std::array<std::uint64_t, 6> typeCounts {};
	for(auto& typeCount : typeCounts) {
		if(!readBinary(input, typeCount)) {
			return false;
		}
	}

	if(magic != kFrameCacheMagic ||
	   version != kCacheVersion ||
	   frameNumber != info.frameNumber ||
	   particleCount != info.particleCount ||
	   particleCount > cacheParticleLimit + 1024) {
		return false;
	}

	if(particles == nullptr) {
		return true;
	}

	particles->clear();
	particles->resize(static_cast<std::size_t>(particleCount));
	for(auto& particle : *particles) {
		float payload[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		input.read(reinterpret_cast<char*>(payload), sizeof(payload));
		if(!input) {
			particles->clear();
			return false;
		}
		particle.position = glm::vec3(payload[0], payload[1], payload[2]);
		particle.nextPosition = particle.position;
		particle.typeIntensity = payload[3];
	}

	return true;
}

bool ofApp::writeCachedFrameData(std::size_t frameIndex, const std::vector<RenderParticle>& particles) const {
	if(frameIndex >= renderFrames.size()) {
		return false;
	}

	std::error_code ec;
	std::filesystem::create_directories(cacheDirectory, ec);
	if(ec) {
		ofLogWarning() << "[collide-vis] Failed to create cache directory: " << cacheDirectory.string();
		return false;
	}

	const auto& info = renderFrames[frameIndex];
	const std::filesystem::path tempPath =
		cacheDirectory / ("frame-" + ofToString(info.frameNumber) + ".tmp");
	const std::filesystem::path framePath =
		cacheDirectory / ("frame-" + ofToString(info.frameNumber) + ".bin");
	std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
	if(!output.is_open()) {
		ofLogWarning() << "[collide-vis] Failed to open frame cache for writing: " << tempPath.string();
		return false;
	}

	const std::uint32_t magic = kFrameCacheMagic;
	const std::uint32_t version = kCacheVersion;
	const std::uint32_t particleCount = static_cast<std::uint32_t>(particles.size());
	if(!writeBinary(output, magic) ||
	   !writeBinary(output, version) ||
	   !writeBinary(output, info.frameNumber) ||
	   !writeBinary(output, info.time) ||
	   !writeBinary(output, particleCount)) {
		return false;
	}
	for(std::size_t typeCount : info.typeCounts) {
		const std::uint64_t value = static_cast<std::uint64_t>(typeCount);
		if(!writeBinary(output, value)) {
			return false;
		}
	}
	for(const auto& particle : particles) {
		const float payload[4] = {
			particle.position.x,
			particle.position.y,
			particle.position.z,
			particle.typeIntensity
		};
		output.write(reinterpret_cast<const char*>(payload), sizeof(payload));
		if(!output) {
			return false;
		}
	}
	output.close();
	if(!output) {
		std::filesystem::remove(tempPath, ec);
		return false;
	}

	std::filesystem::rename(tempPath, framePath, ec);
	if(ec) {
		std::filesystem::remove(framePath, ec);
		ec.clear();
		std::filesystem::rename(tempPath, framePath, ec);
	}
	return !ec;
}

bool ofApp::computeDisplayNormalizationFromCache() {
	if(renderFrames.empty()) {
		return false;
	}

	const std::size_t sampleFrameCount = std::min<std::size_t>(15, renderFrames.size());
	const std::size_t targetSamplesPerFrame = 18000;
	std::vector<glm::vec3> samples;
	samples.reserve(sampleFrameCount * targetSamplesPerFrame);

	for(std::size_t i = 0; i < sampleFrameCount; ++i) {
		const std::size_t frameIndex = sampleFrameCount <= 1
			? 0
			: (i * (renderFrames.size() - 1)) / (sampleFrameCount - 1);
		std::vector<RenderParticle> particles;
		if(!loadCachedFrameData(frameIndex, &particles) || particles.empty()) {
			continue;
		}

		const std::size_t stride = std::max<std::size_t>(1, particles.size() / targetSamplesPerFrame);
		for(std::size_t particleIndex = 0; particleIndex < particles.size(); particleIndex += stride) {
			const auto& particle = particles[particleIndex];
			const int type = static_cast<int>(std::floor(particle.typeIntensity + 0.0001f));
			if(!typeUsedForDisplayNormalization(type)) {
				continue;
			}
			const glm::vec3& position = particle.position;
			if(!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
				continue;
			}
			samples.push_back(position);
		}
	}

	if(samples.size() < 32) {
		return false;
	}

	auto medianComponent = [&samples](int component) {
		std::vector<float> values;
		values.reserve(samples.size());
		for(const auto& sample : samples) {
			values.push_back(sample[component]);
		}
		const std::size_t middle = values.size() / 2;
		std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
		return values[middle];
	};

	const glm::vec3 center(
		medianComponent(0),
		medianComponent(1),
		medianComponent(2)
	);

	std::vector<float> radii;
	radii.reserve(samples.size());
	for(const auto& sample : samples) {
		radii.push_back(glm::length(sample - center));
	}
	std::sort(radii.begin(), radii.end());
	const auto quantile = [&radii](float value) {
		const std::size_t index = std::min<std::size_t>(
			radii.size() - 1,
			static_cast<std::size_t>(std::floor(value * static_cast<float>(radii.size() - 1)))
		);
		return radii[index];
	};

	const float radius50 = quantile(0.50f);
	const float radius90 = quantile(0.90f);
	const float radius95 = quantile(0.95f);
	const float radius99 = quantile(0.99f);
	const float normalizationRadius = std::max(radius95, std::numeric_limits<float>::epsilon());

	displayNormalizationCenter = center;
	displayNormalizationScale = 0.95f / normalizationRadius;
	displayNormalizationReady = true;
	ofLogNotice() << "[collide-vis] Display normalization from cache: samples=" << samples.size()
		<< " center=(" << displayNormalizationCenter.x
		<< ", " << displayNormalizationCenter.y
		<< ", " << displayNormalizationCenter.z
		<< ") r50=" << radius50
		<< " r90=" << radius90
		<< " r95=" << radius95
		<< " r99=" << radius99
		<< " scale=" << displayNormalizationScale;
	return true;
}

void ofApp::warmInitialRamCache(std::size_t startFrameIndex) {
	if(renderFrames.empty() || maxRamCachedFrames == 0) {
		return;
	}

	const std::size_t warmFrameCount = std::min<std::size_t>(24, std::min(maxRamCachedFrames, renderFrames.size()));
	ofLogNotice() << "[collide-vis] Warming RAM frame cache: " << warmFrameCount
		<< " frame(s), about " << ofToString(
			static_cast<double>(warmFrameCount) *
			static_cast<double>(maxParticleCount) *
			static_cast<double>(sizeof(RenderParticle)) /
			(1024.0 * 1024.0),
			0
		) << " MiB";

	const auto warmStart = std::chrono::steady_clock::now();
	for(std::size_t i = 0; i < warmFrameCount; ++i) {
		const std::size_t frameIndex = (startFrameIndex + i) % renderFrames.size();
		if(findRamCachedFrame(frameIndex) != nullptr) {
			continue;
		}
		std::vector<RenderParticle> particles;
		if(!loadCachedFrameData(frameIndex, &particles) || particles.empty()) {
			ofLogWarning() << "[collide-vis] RAM warm skip frame index " << frameIndex;
			continue;
		}
		storeRamCachedFrame(frameIndex, std::move(particles));
		if((i + 1) % 6 == 0 || i + 1 == warmFrameCount) {
			ofLogNotice() << "[collide-vis] RAM warm phase: "
				<< (i + 1) << "/" << warmFrameCount
				<< " cached=" << ramFrameCache.size();
		}
	}

	const float warmMs = std::chrono::duration<float, std::milli>(
		std::chrono::steady_clock::now() - warmStart
	).count();
	ofLogNotice() << "[collide-vis] RAM frame cache ready in " << ofToString(warmMs, 1)
		<< " ms; cached frames=" << ramFrameCache.size();
}

void ofApp::storeRamCachedFrame(std::size_t frameIndex, std::vector<RenderParticle>&& particles) {
	if(frameIndex >= renderFrames.size() || particles.empty() || maxRamCachedFrames == 0) {
		return;
	}

	if(ramFrameCache.find(frameIndex) == ramFrameCache.end()) {
		ramFrameCacheOrder.push_back(frameIndex);
	}
	ramFrameCache[frameIndex] = std::move(particles);

	while(ramFrameCache.size() > maxRamCachedFrames && !ramFrameCacheOrder.empty()) {
		const std::size_t evictIndex = ramFrameCacheOrder.front();
		ramFrameCacheOrder.pop_front();
		if(evictIndex == currentFrameIndex && ramFrameCache.size() <= maxRamCachedFrames + 1) {
			ramFrameCacheOrder.push_back(evictIndex);
			break;
		}
		ramFrameCache.erase(evictIndex);
	}
}

const std::vector<ofApp::RenderParticle>* ofApp::findRamCachedFrame(std::size_t frameIndex) const {
	const auto found = ramFrameCache.find(frameIndex);
	return found == ramFrameCache.end() ? nullptr : &found->second;
}

std::vector<ofApp::RenderParticle> ofApp::buildRenderParticles(
	const SnapshotFrame& frame,
	const glm::vec3& rawCenter,
	float rawScale,
	const std::unordered_map<std::uint64_t, std::size_t>& slotById,
	std::size_t slotCount
) const {
	std::vector<RenderParticle> particles(slotCount);
	if(frame.particles.empty() || slotById.empty() || slotCount == 0) {
		return particles;
	}
	for(auto& particle : particles) {
		particle.typeIntensity = 5.0f;
	}

	std::size_t filledCount = 0;
	for(const auto& source : frame.particles) {
		const auto found = slotById.find(source.id);
		if(found == slotById.end() || found->second >= particles.size()) {
			continue;
		}
		const std::size_t slot = found->second;
		const glm::vec3 currentPosition = (source.position - rawCenter) * rawScale;

		particles[slot].position = currentPosition;
		particles[slot].nextPosition = currentPosition;
		particles[slot].typeIntensity = packedTypeIntensity(source);
		++filledCount;
	}

	if(filledCount == 0) {
		particles.clear();
	}
	return particles;
}

bool ofApp::loadRenderFrame(std::size_t frameIndex) {
	if(frameIndex >= renderFrames.size()) {
		return false;
	}
	if(loadedRenderFrameIndex == frameIndex && frameTexture.isAllocated()) {
		return true;
	}

	if(const auto* cachedParticles = findRamCachedFrame(frameIndex)) {
		uploadCurrentFrame(frameIndex, *cachedParticles);
		return true;
	}

	std::vector<RenderParticle> particles;
	if(!loadCachedFrameData(frameIndex, &particles)) {
		ofLogError() << "[collide-vis] Failed to load cached render frame "
			<< renderFrames[frameIndex].frameNumber;
		return false;
	}

	uploadCurrentFrame(frameIndex, particles);
	storeRamCachedFrame(frameIndex, std::move(particles));
	if(shouldLogFrameNumber(renderFrames[frameIndex].frameNumber)) {
		ofLogNotice() << "[collide-vis] Loaded cached render frame "
			<< renderFrames[frameIndex].frameNumber
			<< " into RAM with " << renderFrames[frameIndex].particleCount << " particles";
	}
	return true;
}

bool ofApp::loadNextRenderFrame(std::size_t frameIndex) {
	if(frameIndex >= renderFrames.size()) {
		return false;
	}
	if(loadedNextRenderFrameIndex == frameIndex && nextFrameTexture.isAllocated()) {
		return true;
	}

	if(const auto* cachedParticles = findRamCachedFrame(frameIndex)) {
		uploadNextFrame(frameIndex, *cachedParticles);
		return true;
	}

	std::vector<RenderParticle> particles;
	if(!loadCachedFrameData(frameIndex, &particles)) {
		ofLogError() << "[collide-vis] Failed to load cached next render frame "
			<< renderFrames[frameIndex].frameNumber;
		return false;
	}

	uploadNextFrame(frameIndex, particles);
	storeRamCachedFrame(frameIndex, std::move(particles));
	return true;
}

bool ofApp::loadAheadRenderFrame(std::size_t frameIndex) {
	if(frameIndex >= renderFrames.size()) {
		return false;
	}
	if(loadedAheadRenderFrameIndex == frameIndex && aheadFrameTexture.isAllocated()) {
		return true;
	}

	if(const auto* cachedParticles = findRamCachedFrame(frameIndex)) {
		uploadAheadFrame(frameIndex, *cachedParticles);
		return true;
	}

	std::vector<RenderParticle> particles;
	if(!loadCachedFrameData(frameIndex, &particles)) {
		ofLogError() << "[collide-vis] Failed to load cached ahead render frame "
			<< renderFrames[frameIndex].frameNumber;
		return false;
	}

	uploadAheadFrame(frameIndex, particles);
	storeRamCachedFrame(frameIndex, std::move(particles));
	return true;
}

void ofApp::uploadCurrentFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles) {
	populateTextureFromParticles(frameTexture, frameUploadPixels, frameUploadedTexelCount, particles, true);
	updateInitialCameraFocusFromParticles(particles);
	loadedRenderFrameIndex = frameIndex;
}

void ofApp::uploadNextFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles) {
	populateTextureFromParticles(nextFrameTexture, nextFrameUploadPixels, nextFrameUploadedTexelCount, particles, true);
	loadedNextRenderFrameIndex = frameIndex;
}

void ofApp::uploadAheadFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles) {
	populateTextureFromParticles(aheadFrameTexture, aheadFrameUploadPixels, aheadFrameUploadedTexelCount, particles, true);
	loadedAheadRenderFrameIndex = frameIndex;
}

void ofApp::populateTextureFromParticles(
	ofTexture& texture,
	ofFloatPixels& pixels,
	std::size_t& uploadedTexelCount,
	const std::vector<RenderParticle>& particles,
	bool clearUnusedTexels
) {
	bool allocatedPixels = false;
	if(pixels.getWidth() != textureWidth ||
	   pixels.getHeight() != textureHeight ||
	   pixels.getNumChannels() != 4) {
		pixels.allocate(textureWidth, textureHeight, OF_PIXELS_RGBA);
		uploadedTexelCount = 0;
		allocatedPixels = true;
	}

	float* data = pixels.getData();
	const std::size_t textureTexelCount =
		static_cast<std::size_t>(textureWidth) * static_cast<std::size_t>(textureHeight);
	const std::size_t count = std::min<std::size_t>(
		particles.size(),
		textureTexelCount
	);
	if(clearUnusedTexels && allocatedPixels) {
		std::fill(data, data + textureTexelCount * 4, 0.0f);
	} else if(clearUnusedTexels && count < uploadedTexelCount) {
		std::fill(data + count * 4, data + uploadedTexelCount * 4, 0.0f);
	}

	for(std::size_t i = 0; i < count; ++i) {
		const std::size_t offset = i * 4;
		const glm::vec3& position = particles[i].position;
		data[offset + 0] = position.x;
		data[offset + 1] = position.y;
		data[offset + 2] = position.z;
		data[offset + 3] = particles[i].typeIntensity;
	}
	uploadedTexelCount = count;

	if(!texture.isAllocated() ||
	   texture.getWidth() != textureWidth ||
	   texture.getHeight() != textureHeight) {
		texture.allocate(textureWidth, textureHeight, GL_RGBA32F);
		texture.setTextureMinMagFilter(GL_NEAREST, GL_NEAREST);
	}
	texture.loadData(pixels);
}

void ofApp::rebuildDrawMesh(std::size_t particleCount) {
	drawMesh.clear();
	drawMesh.setMode(OF_PRIMITIVE_POINTS);
	drawMesh.getVertices().resize(particleCount, glm::vec3(0.0f));
}

void ofApp::syncFrameState() {
	if(renderFrames.empty() || currentFrameIndex >= renderFrames.size()) {
		currentParticleCount = 0;
		currentDisplayParticleCount = 0;
		return;
	}

	if(currentFrameIndex == loadedNextRenderFrameIndex && nextFrameTexture.isAllocated()) {
		std::swap(frameTexture, nextFrameTexture);
		loadedRenderFrameIndex = currentFrameIndex;
		loadedNextRenderFrameIndex = std::numeric_limits<std::size_t>::max();
	}

	if(!loadRenderFrame(currentFrameIndex)) {
		currentParticleCount = 0;
		currentDisplayParticleCount = 0;
		return;
	}

	const std::size_t nextFrameIndex = (currentFrameIndex + 1) % renderFrames.size();
	if(renderFrames.size() > 1) {
		if(nextFrameIndex == loadedAheadRenderFrameIndex && aheadFrameTexture.isAllocated()) {
			std::swap(nextFrameTexture, aheadFrameTexture);
			loadedNextRenderFrameIndex = nextFrameIndex;
			loadedAheadRenderFrameIndex = std::numeric_limits<std::size_t>::max();
		}
		if(loadedNextRenderFrameIndex != nextFrameIndex || !nextFrameTexture.isAllocated()) {
			if(!loadNextRenderFrame(nextFrameIndex)) {
				nextFrameTexture.clear();
				loadedNextRenderFrameIndex = std::numeric_limits<std::size_t>::max();
			}
		}
		beginNextFramePrefetch((nextFrameIndex + 1) % renderFrames.size());
		if(!playing && renderFrames.size() > 2) {
			loadAheadRenderFrame((nextFrameIndex + 1) % renderFrames.size());
		}
	} else {
		nextFrameTexture = frameTexture;
		loadedNextRenderFrameIndex = currentFrameIndex;
	}

	currentParticleCount = renderFrames[currentFrameIndex].particleCount;
	const std::size_t renderParticleCount = getInterpolatedParticleCount();
	displayStride = computeDisplayStride(renderParticleCount);
	const std::size_t nextDisplayParticleCount = getDisplayParticleCount(renderParticleCount);
	if(nextDisplayParticleCount != currentDisplayParticleCount) {
		currentDisplayParticleCount = nextDisplayParticleCount;
		rebuildDrawMesh(currentDisplayParticleCount);
	}
}

void ofApp::prepareAheadFrameTexture() {
	if(!playing || renderFrames.size() <= 2 || currentFrameIndex >= renderFrames.size()) {
		return;
	}

	const std::size_t nextFrameIndex = (currentFrameIndex + 1) % renderFrames.size();
	const std::size_t aheadFrameIndex = (nextFrameIndex + 1) % renderFrames.size();
	if(loadedAheadRenderFrameIndex == aheadFrameIndex && aheadFrameTexture.isAllocated()) {
		return;
	}

	beginNextFramePrefetch(aheadFrameIndex);

	// Move the expensive GPU upload away from the frame boundary. At the boundary
	// syncFrameState can then promote next/ahead textures with swaps only.
	if(currentFrameBlend < 0.32f || currentFrameBlend > 0.72f) {
		return;
	}

	const auto* cachedParticles = findRamCachedFrame(aheadFrameIndex);
	if(cachedParticles == nullptr) {
		return;
	}
	uploadAheadFrame(aheadFrameIndex, *cachedParticles);
	beginNextFramePrefetch((aheadFrameIndex + 1) % renderFrames.size());
}

void ofApp::beginNextFramePrefetch(std::size_t frameIndex) {
	if(renderFrames.size() <= 1 || frameIndex >= renderFrames.size()) {
		return;
	}
	if(nextFramePrefetchFuture.valid()) {
		const auto status = nextFramePrefetchFuture.wait_for(std::chrono::milliseconds(0));
		if(status != std::future_status::ready) {
			return;
		}
		PrefetchedFrameData prefetched = nextFramePrefetchFuture.get();
		if(prefetched.frameIndex != std::numeric_limits<std::size_t>::max() && !prefetched.particles.empty()) {
			storeRamCachedFrame(prefetched.frameIndex, std::move(prefetched.particles));
		}
	}

	std::size_t targetFrameIndex = frameIndex;
	const std::size_t lookAheadCount = std::min<std::size_t>(maxRamCachedFrames, renderFrames.size());
	bool foundTarget = false;
	for(std::size_t offset = 0; offset < lookAheadCount; ++offset) {
		const std::size_t candidate = (frameIndex + offset) % renderFrames.size();
		if(candidate == loadedRenderFrameIndex || findRamCachedFrame(candidate) != nullptr) {
			continue;
		}
		targetFrameIndex = candidate;
		foundTarget = true;
		break;
	}
	if(!foundTarget) {
		return;
	}

	nextFramePrefetchFuture = std::async(std::launch::async, [this, targetFrameIndex]() {
		PrefetchedFrameData prefetched;
		prefetched.frameIndex = targetFrameIndex;
		if(!loadCachedFrameData(targetFrameIndex, &prefetched.particles)) {
			prefetched.frameIndex = std::numeric_limits<std::size_t>::max();
			prefetched.particles.clear();
		}
		return prefetched;
	});
}

void ofApp::finishNextFramePrefetch() {
	if(!nextFramePrefetchFuture.valid()) {
		return;
	}
	const auto status = nextFramePrefetchFuture.wait_for(std::chrono::milliseconds(0));
	if(status != std::future_status::ready) {
		return;
	}

	PrefetchedFrameData prefetched = nextFramePrefetchFuture.get();
	if(prefetched.frameIndex == std::numeric_limits<std::size_t>::max() || prefetched.particles.empty()) {
		return;
	}
	storeRamCachedFrame(prefetched.frameIndex, std::move(prefetched.particles));
}

void ofApp::waitForPendingPrefetch() {
	if(!nextFramePrefetchFuture.valid()) {
		return;
	}
	nextFramePrefetchFuture.wait();
	try {
		(void)nextFramePrefetchFuture.get();
	} catch(...) {
	}
}

void ofApp::updatePlaybackPosition(double deltaSeconds) {
	if(renderFrames.empty()) {
		return;
	}
	deltaSeconds = std::min(deltaSeconds, 1.0 / 30.0);
	const double interval = 1.0 / static_cast<double>(std::max(0.001f, playbackFpsParam.get()));
	playbackFramePosition += deltaSeconds / interval;
	const double frameCount = static_cast<double>(renderFrames.size());
	playbackFramePosition = std::fmod(playbackFramePosition, frameCount);
	if(playbackFramePosition < 0.0) {
		playbackFramePosition += frameCount;
	}
}

void ofApp::applyPlaybackPosition(bool reloadRenderFrame) {
	if(renderFrames.empty()) {
		return;
	}

	std::size_t nextFrameIndex = currentFrameIndex;
	float nextBlend = currentFrameBlend;
	frameStateForPosition(playbackFramePosition, nextFrameIndex, nextBlend);
	const bool frameChanged = nextFrameIndex != currentFrameIndex;
	currentFrameIndex = nextFrameIndex;
	currentFrameBlend = nextBlend;
	if(reloadRenderFrame && (frameChanged || !frameTexture.isAllocated())) {
		syncFrameState();
	}
}

void ofApp::frameStateForPosition(double playbackPosition, std::size_t& frameIndex, float& frameBlend) const {
	frameIndex = 0;
	frameBlend = 0.0f;
	if(renderFrames.empty()) {
		return;
	}

	const double frameCount = static_cast<double>(renderFrames.size());
	double wrapped = std::fmod(playbackPosition, frameCount);
	if(wrapped < 0.0) {
		wrapped += frameCount;
	}
	frameIndex = static_cast<std::size_t>(std::floor(wrapped)) % renderFrames.size();
	frameBlend = static_cast<float>(wrapped - std::floor(wrapped));
}

std::size_t ofApp::getInterpolatedParticleCount() const {
	if(renderFrames.empty() || currentFrameIndex >= renderFrames.size()) {
		return 0;
	}
	return currentParticleCount;
}

std::size_t ofApp::getDisplayParticleCount(std::size_t particleCount) const {
	if(particleCount == 0) {
		return 0;
	}
	return (particleCount + displayStride - 1) / displayStride;
}

bool ofApp::particleTypeEnabled(int type) const {
	switch(type) {
		case 0: return showGasParam;
		case 1: return showDarkMatterParam;
		case 2: return showDiskParam;
		case 3: return showBulgeParam;
		case 4: return showStarsParam;
		default: return true;
	}
}

float ofApp::packedTypeIntensity(const SnapshotParticle& particle) const {
	return packedTypeIntensityValue(particle.type, particle.intensity);
}

void ofApp::syncDisplaySettings() {
	const std::size_t renderParticleCount = getInterpolatedParticleCount();
	const std::size_t desiredStride = renderParticleCount == 0
		? 1
		: computeDisplayStride(renderParticleCount);
	if(maxDrawCountParam.get() == previousMaxDrawCount &&
	   fullResolutionParam.get() == previousFullResolution &&
	   autoLodParam.get() == previousAutoLod &&
	   desiredStride == displayStride) {
		return;
	}
	previousMaxDrawCount = maxDrawCountParam;
	previousFullResolution = fullResolutionParam;
	previousAutoLod = autoLodParam;

	displayStride = desiredStride;
	const std::size_t nextDisplayParticleCount = getDisplayParticleCount(renderParticleCount);
	if(nextDisplayParticleCount != currentDisplayParticleCount) {
		currentDisplayParticleCount = nextDisplayParticleCount;
		rebuildDrawMesh(currentDisplayParticleCount);
	}
}

std::size_t ofApp::computeDisplayStride(std::size_t renderParticleCount) const {
	if(renderParticleCount == 0) {
		return 1;
	}

	const std::size_t drawCap = static_cast<std::size_t>(std::max(1, maxDrawCountParam.get()));
	if(!fullResolutionParam) {
		return std::max<std::size_t>(1, (renderParticleCount + drawCap - 1) / drawCap);
	}

	if(!autoLodParam) {
		return 1;
	}

	const float sceneScale = std::max(sceneScaleParam.get(), 0.001f);
	const float viewScale = std::max(coreScaleParam.get(), 0.001f);
	const float normalizedRadius = initialCameraFocusValid
		? initialCameraRadius * (displayNormalizationReady ? displayNormalizationScale : 1.0f)
		: 0.75f;
	const float sceneRadius = std::max(normalizedRadius * sceneScale * viewScale, 0.08f);
	const float distance = std::max(cam.getDistance(), 0.001f);
	const float nearDistance = std::max(sceneRadius * 1.45f, kCameraInitialMinDistance);
	const float farDistance = std::max(sceneRadius * 3.0f, nearDistance + 0.001f);
	float lod = ofClamp((distance - nearDistance) / (farDistance - nearDistance), 0.0f, 1.0f);
	lod = lod * lod * (3.0f - 2.0f * lod);
	if(lod <= 0.001f) {
		return 1;
	}

	const float farDrawCount = static_cast<float>(std::min<std::size_t>(renderParticleCount, drawCap));
	const float targetDrawCount = ofLerp(static_cast<float>(renderParticleCount), farDrawCount, lod);
	return std::max<std::size_t>(
		1,
		(renderParticleCount + static_cast<std::size_t>(std::max(1.0f, targetDrawCount)) - 1) /
		static_cast<std::size_t>(std::max(1.0f, targetDrawCount))
	);
}

void ofApp::updateInitialCameraFocusFromParticles(const std::vector<RenderParticle>& particles) {
	if(initialCameraFocusValid || particles.empty()) {
		return;
	}

	auto computeFocus = [&particles](bool excludeDarkMatter, glm::vec3& focus, float& radius) {
		glm::dvec3 sum(0.0);
		std::size_t count = 0;
		for(const auto& particle : particles) {
			const int type = static_cast<int>(std::floor(particle.typeIntensity + 0.0001f));
			if(excludeDarkMatter && type == 1) {
				continue;
			}
			const glm::vec3& position = particle.position;
			if(!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
				continue;
			}
			sum += glm::dvec3(position);
			++count;
		}

		if(count == 0) {
			return false;
		}

		focus = glm::vec3(sum / static_cast<double>(count));
		double sumRadiusSquared = 0.0;
		for(const auto& particle : particles) {
			const int type = static_cast<int>(std::floor(particle.typeIntensity + 0.0001f));
			if(excludeDarkMatter && type == 1) {
				continue;
			}
			const glm::vec3& position = particle.position;
			if(!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
				continue;
			}
			const glm::vec3 delta = position - focus;
			sumRadiusSquared += static_cast<double>(glm::dot(delta, delta));
		}

		radius = std::sqrt(static_cast<float>(sumRadiusSquared / static_cast<double>(count)));
		return radius > std::numeric_limits<float>::epsilon();
	};

	glm::vec3 focus(0.0f);
	float radius = 1.0f;
	if(!computeFocus(true, focus, radius) && !computeFocus(false, focus, radius)) {
		return;
	}

	initialCameraFocus = focus;
	initialCameraRadius = std::max(radius, 0.05f);
	initialCameraFocusValid = true;
	ofLogNotice() << "[collide-vis] Initial camera focus "
		<< initialCameraFocus.x << ", "
		<< initialCameraFocus.y << ", "
		<< initialCameraFocus.z
		<< " radius=" << initialCameraRadius;
}

glm::vec3 ofApp::transformCachedPositionForDisplay(const glm::vec3& position) const {
	if(!displayNormalizationReady) {
		return position;
	}
	return (position - displayNormalizationCenter) * displayNormalizationScale;
}

glm::vec3 ofApp::transformCachedPositionForPtsm(const glm::vec3& position) const {
	return transformCachedPositionForDisplay(position) *
		std::max(sceneScaleParam.get(), 0.001f) *
		std::max(coreScaleParam.get(), 0.001f);
}

void ofApp::getVisibleBounds(glm::vec3& boundsMin, glm::vec3& boundsMax) const {
	const float scale = std::max(sceneScaleParam.get(), 0.001f) *
		std::max(coreScaleParam.get(), 0.001f);
	boundsMin = worldMin * scale;
	boundsMax = worldMax * scale;
}

void ofApp::resetCameraView() {
	cam.setAutoDistance(false);
	const float sceneScale = std::max(sceneScaleParam.get(), 0.001f);
	const float viewScale = std::max(coreScaleParam.get(), 0.001f);
	const glm::vec3 target = initialCameraFocusValid
		? transformCachedPositionForDisplay(initialCameraFocus) * sceneScale * viewScale
		: glm::vec3(0.0f);
	const float normalizedRadius = initialCameraRadius *
		(displayNormalizationReady ? displayNormalizationScale : 1.0f);
	const float distance = initialCameraFocusValid
		? ofClamp(
			normalizedRadius * sceneScale * viewScale * kCameraFocusDistanceFactor,
			kCameraInitialMinDistance,
			kCameraInitialMaxDistance
		)
		: kCameraInitialDistance;
	const glm::vec3 viewDirection = glm::normalize(glm::vec3(0.42f, -0.18f, 1.0f));

	cam.setTarget(target);
	cam.setPosition(target + viewDirection * distance);
	cam.lookAt(target, glm::vec3(0.0f, 1.0f, 0.0f));
	cam.setDistance(distance);
	cam.setNearClip(kCameraNearClip);
	cam.setFarClip(kCameraFarClip);
	cam.enableMouseInput();
}

void ofApp::constrainCameraDistance() {
	const float distance = cam.getDistance();
	if(distance < kCameraDollyDistance) {
		const glm::vec3 zAxis = cam.getZAxis();
		if(glm::length(zAxis) > std::numeric_limits<float>::epsilon()) {
			const glm::vec3 forward = -glm::normalize(zAxis);
			const glm::vec3 targetPosition = cam.getTarget().getPosition();
			cam.setTarget(targetPosition + forward * (kCameraDollyDistance - distance));
		}
		cam.setDistance(kCameraDollyDistance);
	} else if(distance > kCameraMaxDistance) {
		cam.setDistance(kCameraMaxDistance);
	}
	cam.setNearClip(kCameraNearClip);
	cam.setFarClip(kCameraFarClip);
}

void ofApp::syncPtsmSettings() {
	if(!ptsmGui.enabled) {
		return;
	}

	ptsmGui.applyTo(ptsmSettings);
	ptsmSettings.field.normalizeByAttractorCount = true;
	ptsmField.setSigma(ptsmSettings.field.sigma);
	ptsmField.setFieldGain(ptsmSettings.field.fieldGain * ofClamp(ptsmDensityMixParam.get(), 0.0f, 1.0f));
	ptsmField.setNormalizeByAttractorCount(ptsmSettings.field.normalizeByAttractorCount);

	ptsm::ProbeEngineConfig config;
	config.mass = ptsmSettings.probe.mass;
	config.energyRetention = ptsmSettings.probe.energyRetention;
	config.substeps = ptsmSettings.probe.substeps;
	config.trailLength = ptsmSettings.probe.trailLength;
	config.boundsEnabled = ptsmSettings.probe.boundsEnabled;
	getVisibleBounds(config.boundsMin, config.boundsMax);
	config.boundaryBounce = ptsmSettings.probe.boundaryBounce;
	config.maxForce = ptsmSettings.probe.maxForce;
	config.maxSpeed = ptsmSettings.probe.maxSpeed;
	ptsmEngine.setConfig(config);

	const float newProbeRadius = ptsmProbeRadiusParam;
	const float newTrailAlpha = ptsmTrailAlphaParam;
	const int newTrailSmoothing = ptsmTrailSmoothingParam;
	if(std::abs(ptsmProbeRadius - newProbeRadius) > 0.000001f) {
		ptsmProbeRadius = newProbeRadius;
		ptsmProbeSphere.setRadius(ptsmProbeRadius);
	}
	ptsmTrailAlpha = newTrailAlpha;
	if(ptsmTrailSmoothingSize != newTrailSmoothing) {
		ptsmTrailSmoothingSize = newTrailSmoothing;
		ptsmTrailCachedSize = 0;
	}

	if(ptsmSettings.osc.enabled && !ptsmOscReady) {
		ptsmOscSender.setup(ptsmSettings.osc.host, ptsmSettings.osc.port);
		ptsmOscReady = true;
	}
}

void ofApp::rebuildPtsmFieldIfNeeded() {
	if(!ptsmGui.enabled || renderFrames.empty() || currentFrameIndex >= renderFrames.size()) {
		return;
	}

	const int sampleCount = std::max(1, ptsmGui.sampleCount.get());
	const float sceneScale = sceneScaleParam.get();
	const float viewScale = coreScaleParam.get();
	const bool spawnSettingsChanged =
		ptsmLastSampleCount >= 0 &&
		(ptsmLastSampleCount != sampleCount ||
		 std::abs(ptsmLastSceneScale - sceneScale) > 0.0001f ||
		 std::abs(ptsmLastViewScale - viewScale) > 0.0001f);
	if(ptsmFieldFrameIndex == currentFrameIndex &&
	   ptsmLastSampleCount == sampleCount &&
	   std::abs(ptsmLastSceneScale - sceneScale) <= 0.0001f &&
	   std::abs(ptsmLastViewScale - viewScale) <= 0.0001f) {
		ptsmEngine.setFramePosition(currentFrameBlend);
		return;
	}
	if(spawnSettingsChanged) {
		ptsmCurrentSpawnPosition.reset();
		ptsmCurrentSpawnVelocity.reset();
		ptsmLockedSpawnPosition.reset();
		ptsmLockedSpawnVelocity.reset();
	}

	std::vector<glm::vec3> currentAttractors;
	std::vector<glm::vec3> nextAttractors;
	std::vector<PtsmFlowSample> currentFlowSamples;
	std::vector<PtsmFlowSample> nextFlowSamples;
	if(!buildPtsmFrame(currentFrameIndex, currentAttractors) || currentAttractors.empty()) {
		return;
	}
	buildPtsmFlowFrame(currentFrameIndex, currentFlowSamples);

	const std::size_t nextFrameIndex = renderFrames.size() > 1
		? (currentFrameIndex + 1) % renderFrames.size()
		: currentFrameIndex;
	if(!buildPtsmFrame(nextFrameIndex, nextAttractors) || nextAttractors.empty()) {
		nextAttractors = currentAttractors;
	}
	if(!buildPtsmFlowFrame(nextFrameIndex, nextFlowSamples) || nextFlowSamples.empty()) {
		nextFlowSamples = currentFlowSamples;
	}

	ptsmField.setFrames({currentAttractors, nextAttractors});
	ptsmFlowSamplesCurrent = std::move(currentFlowSamples);
	ptsmFlowSamplesNext = std::move(nextFlowSamples);
	ptsmEngine.setFramePosition(currentFrameBlend);
	ptsmFieldFrameIndex = currentFrameIndex;
	ptsmLastSampleCount = sampleCount;
	ptsmLastSceneScale = sceneScale;
	ptsmLastViewScale = viewScale;
}

bool ofApp::buildPtsmFrame(std::size_t frameIndex, std::vector<glm::vec3>& attractors) const {
	attractors.clear();
	if(frameIndex >= renderFrames.size()) {
		return false;
	}

	std::vector<RenderParticle> particles;
	if(const auto* cached = findRamCachedFrame(frameIndex)) {
		particles = *cached;
	} else if(!loadCachedFrameData(frameIndex, &particles)) {
		return false;
	}

	std::vector<glm::vec3> candidates;
	candidates.reserve(particles.size());
	for(const auto& particle : particles) {
		const int type = static_cast<int>(std::floor(particle.typeIntensity + 0.0001f));
		if(!typeUsedForDisplayNormalization(type) || !particleTypeEnabled(type)) {
			continue;
		}
		const glm::vec3 position = transformCachedPositionForPtsm(particle.position);
		if(std::isfinite(position.x) && std::isfinite(position.y) && std::isfinite(position.z)) {
			candidates.push_back(position);
		}
	}

	if(candidates.empty()) {
		candidates.reserve(particles.size());
		for(const auto& particle : particles) {
			const glm::vec3 position = transformCachedPositionForPtsm(particle.position);
			if(std::isfinite(position.x) && std::isfinite(position.y) && std::isfinite(position.z)) {
				candidates.push_back(position);
			}
		}
	}

	attractors = compressPtsmAttractors(candidates, std::max(1, ptsmGui.sampleCount.get()));
	return !attractors.empty();
}

std::vector<glm::vec3> ofApp::compressPtsmAttractors(const std::vector<glm::vec3>& points, int maxAttractors) const {
	if(points.empty()) {
		return {};
	}
	if(static_cast<int>(points.size()) <= maxAttractors) {
		return points;
	}

	glm::vec3 srcMin(std::numeric_limits<float>::max());
	glm::vec3 srcMax(std::numeric_limits<float>::lowest());
	for(const auto& point : points) {
		srcMin = glm::min(srcMin, point);
		srcMax = glm::max(srcMax, point);
	}

	const glm::vec3 srcSize = glm::max(srcMax - srcMin, glm::vec3(std::numeric_limits<float>::epsilon()));
	int gridResolution = 8;
	std::vector<glm::vec3> compressed;

	while(true) {
		std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> voxels;
		voxels.reserve(points.size() / 2);

		for(const auto& point : points) {
			const glm::vec3 normalized = (point - srcMin) / srcSize;
			const VoxelKey key = {
				std::min(gridResolution - 1, std::max(0, static_cast<int>(normalized.x * gridResolution))),
				std::min(gridResolution - 1, std::max(0, static_cast<int>(normalized.y * gridResolution))),
				std::min(gridResolution - 1, std::max(0, static_cast<int>(normalized.z * gridResolution)))
			};
			auto& voxel = voxels[key];
			voxel.sum += point;
			++voxel.count;
		}

		compressed.clear();
		compressed.reserve(voxels.size());
		for(const auto& entry : voxels) {
			const VoxelAccumulator& voxel = entry.second;
			compressed.push_back(voxel.sum / static_cast<float>(std::max(1, voxel.count)));
		}

		if(static_cast<int>(compressed.size()) <= maxAttractors || gridResolution <= 2) {
			break;
		}
		gridResolution = std::max(2, gridResolution - 1);
	}

	return compressed;
}

bool ofApp::buildPtsmFlowFrame(std::size_t frameIndex, std::vector<PtsmFlowSample>& samples) const {
	samples.clear();
	if(frameIndex >= renderFrames.size()) {
		return false;
	}

	std::vector<RenderParticle> particles;
	if(const auto* cached = findRamCachedFrame(frameIndex)) {
		particles = *cached;
	} else if(!loadCachedFrameData(frameIndex, &particles)) {
		return false;
	}

	const std::size_t nextFrameIndex = renderFrames.size() > 1
		? (frameIndex + 1) % renderFrames.size()
		: frameIndex;
	std::vector<RenderParticle> nextParticles;
	if(const auto* cached = findRamCachedFrame(nextFrameIndex)) {
		nextParticles = *cached;
	} else if(!loadCachedFrameData(nextFrameIndex, &nextParticles)) {
		return false;
	}

	std::vector<PtsmFlowSample> candidates;
	candidates.reserve(particles.size());
	const std::size_t count = std::min(particles.size(), nextParticles.size());
	glm::vec3 meanVelocity(0.0f);
	for(std::size_t i = 0; i < count; ++i) {
		const auto& particle = particles[i];
		const int type = static_cast<int>(std::floor(particle.typeIntensity + 0.0001f));
		if(!typeUsedForDisplayNormalization(type) || !particleTypeEnabled(type)) {
			continue;
		}

		const glm::vec3 position = transformCachedPositionForPtsm(particle.position);
		const glm::vec3 nextPosition = transformCachedPositionForPtsm(nextParticles[i].position);
		const glm::vec3 velocity = nextPosition - position;
		if(std::isfinite(position.x) && std::isfinite(position.y) && std::isfinite(position.z) &&
		   std::isfinite(velocity.x) && std::isfinite(velocity.y) && std::isfinite(velocity.z)) {
			candidates.push_back({position, velocity});
			meanVelocity += velocity;
		}
	}
	if(!candidates.empty()) {
		meanVelocity /= static_cast<float>(candidates.size());
		const float verticalMix = ofClamp(ptsmFlowVerticalMixParam.get(), 0.0f, 1.0f);
		for(auto& sample : candidates) {
			sample.velocity -= meanVelocity;
			sample.velocity.y *= verticalMix;
		}
	}

	samples = compressPtsmFlowSamples(candidates, std::max(1, ptsmGui.sampleCount.get()));
	return !samples.empty();
}

std::vector<ofApp::PtsmFlowSample> ofApp::compressPtsmFlowSamples(const std::vector<PtsmFlowSample>& samples, int maxSamples) const {
	if(samples.empty()) {
		return {};
	}
	if(static_cast<int>(samples.size()) <= maxSamples) {
		return samples;
	}

	glm::vec3 srcMin(std::numeric_limits<float>::max());
	glm::vec3 srcMax(std::numeric_limits<float>::lowest());
	for(const auto& sample : samples) {
		srcMin = glm::min(srcMin, sample.position);
		srcMax = glm::max(srcMax, sample.position);
	}

	const glm::vec3 srcSize = glm::max(srcMax - srcMin, glm::vec3(std::numeric_limits<float>::epsilon()));
	int gridResolution = 8;
	std::vector<PtsmFlowSample> compressed;

	while(true) {
		std::unordered_map<VoxelKey, FlowVoxelAccumulator, VoxelKeyHash> voxels;
		voxels.reserve(samples.size() / 2);

		for(const auto& sample : samples) {
			const glm::vec3 normalized = (sample.position - srcMin) / srcSize;
			const VoxelKey key = {
				std::min(gridResolution - 1, std::max(0, static_cast<int>(normalized.x * gridResolution))),
				std::min(gridResolution - 1, std::max(0, static_cast<int>(normalized.y * gridResolution))),
				std::min(gridResolution - 1, std::max(0, static_cast<int>(normalized.z * gridResolution)))
			};
			auto& voxel = voxels[key];
			voxel.positionSum += sample.position;
			voxel.velocitySum += sample.velocity;
			++voxel.count;
		}

		compressed.clear();
		compressed.reserve(voxels.size());
		for(const auto& entry : voxels) {
			const FlowVoxelAccumulator& voxel = entry.second;
			const float count = static_cast<float>(std::max(1, voxel.count));
			compressed.push_back({
				voxel.positionSum / count,
				voxel.velocitySum / count
			});
		}

		if(static_cast<int>(compressed.size()) <= maxSamples || gridResolution <= 2) {
			break;
		}
		gridResolution = std::max(2, gridResolution - 1);
	}

	return compressed;
}

ofApp::GalaxyFieldLocal ofApp::sampleGalaxyField(const glm::vec3& position, const glm::vec3& probeVelocity) const {
	GalaxyFieldLocal current = sampleGalaxyFieldFromFrame(ptsmFlowSamplesCurrent, position, probeVelocity);
	if(ptsmFlowSamplesNext.empty()) {
		return current;
	}

	const GalaxyFieldLocal next = sampleGalaxyFieldFromFrame(ptsmFlowSamplesNext, position, probeVelocity);
	const float blend = ofClamp(currentFrameBlend, 0.0f, 1.0f);
	GalaxyFieldLocal result;
	result.flowVelocity = glm::mix(current.flowVelocity, next.flowVelocity, blend);
	result.gradU = current.gradU * (1.0f - blend) + next.gradU * blend;
	result.vorticity = glm::mix(current.vorticity, next.vorticity, blend);
	result.density = glm::mix(current.density, next.density, blend);
	result.dispersion = glm::mix(current.dispersion, next.dispersion, blend);
	result.divergence = glm::mix(current.divergence, next.divergence, blend);
	result.compression = std::max(0.0f, -result.divergence);
	result.shear = glm::mix(current.shear, next.shear, blend);
	result.slip = glm::length(result.flowVelocity - probeVelocity);
	result.vorticityMag = glm::length(result.vorticity);
	result.valid = current.valid || next.valid;
	return result;
}

ofApp::GalaxyFieldLocal ofApp::sampleGalaxyFieldFromFrame(
	const std::vector<PtsmFlowSample>& samples,
	const glm::vec3& position,
	const glm::vec3& probeVelocity
) const {
	GalaxyFieldLocal result;
	if(samples.empty()) {
		return result;
	}

	const float radius = std::max(ptsmFlowRadiusParam.get(), std::numeric_limits<float>::epsilon());
	const float denom = 2.0f * radius * radius;
	glm::vec3 weightedVelocity(0.0f);
	float totalWeight = 0.0f;
	for(const auto& sample : samples) {
		const glm::vec3 delta = position - sample.position;
		const float squaredDistance = glm::dot(delta, delta);
		const float weight = std::exp(-squaredDistance / denom);
		weightedVelocity += sample.velocity * weight;
		totalWeight += weight;
	}

	if(totalWeight <= std::numeric_limits<float>::epsilon()) {
		return result;
	}

	result.flowVelocity = weightedVelocity / totalWeight;
	result.density = totalWeight / static_cast<float>(std::max<std::size_t>(1, samples.size()));

	glm::mat3 crr(0.0f);
	glm::mat3 cvr(0.0f);
	float dispersionSum = 0.0f;
	for(const auto& sample : samples) {
		const glm::vec3 r = sample.position - position;
		const glm::vec3 dv = sample.velocity - result.flowVelocity;
		const float squaredDistance = glm::dot(r, r);
		const float weight = std::exp(-squaredDistance / denom);
		crr += outerProduct(r, r) * weight;
		cvr += outerProduct(dv, r) * weight;
		dispersionSum += glm::dot(dv, dv) * weight;
	}

	const float regularization = std::max(0.0001f, radius * radius * 0.0001f);
	result.gradU = cvr * glm::inverse(crr + glm::mat3(regularization));
	result.dispersion = std::sqrt(std::max(0.0f, dispersionSum / totalWeight));
	result.divergence = result.gradU[0][0] + result.gradU[1][1] + result.gradU[2][2];
	result.compression = std::max(0.0f, -result.divergence);
	result.vorticity = glm::vec3(
		result.gradU[2][1] - result.gradU[1][2],
		result.gradU[0][2] - result.gradU[2][0],
		result.gradU[1][0] - result.gradU[0][1]
	);
	result.vorticityMag = glm::length(result.vorticity);

	const glm::mat3 strain = (result.gradU + glm::transpose(result.gradU)) * 0.5f;
	const glm::mat3 deviatoric = strain - glm::mat3(result.divergence / 3.0f);
	result.shear = frobeniusNorm(deviatoric);
	result.slip = glm::length(result.flowVelocity - probeVelocity);
	result.valid = true;
	return result;
}

void ofApp::applyPtsmFlowForces(float dt) {
	if(ptsmFlowCouplingParam <= 0.0f || ptsmEngine.getProbeCount() == 0) {
		ptsmLastGalaxyField = GalaxyFieldLocal();
		return;
	}

	ptsm::ProbeState* probe = ptsmEngine.getMutableProbeState(0);
	if(probe == nullptr) {
		ptsmLastGalaxyField = GalaxyFieldLocal();
		return;
	}

	GalaxyFieldLocal local = sampleGalaxyField(probe->position, probe->velocity);
	if(!local.valid) {
		ptsmLastGalaxyField = local;
		return;
	}

	const float fieldScale = std::max(0.0f, ptsmFlowVelocityScaleParam.get());
	const glm::vec3 localFlowVelocity = local.flowVelocity * fieldScale;
	const glm::vec3 slipVector = localFlowVelocity - probe->velocity;
	const glm::vec3 slipDirection = glm::length(slipVector) > std::numeric_limits<float>::epsilon()
		? glm::normalize(slipVector)
		: randomUnitVector();
	const glm::vec3 velocityDirection = glm::length(probe->velocity) > std::numeric_limits<float>::epsilon()
		? glm::normalize(probe->velocity)
		: slipDirection;
	const glm::vec3 desiredAcceleration =
		slipVector *
		std::max(0.0f, ptsmFlowCouplingParam.get());
	const glm::vec3 tidalAcceleration =
		(local.gradU * probe->velocity) *
		fieldScale *
		std::max(0.0f, ptsmTidalGainParam.get());
	glm::vec3 vortexAxis = local.vorticity;
	if(glm::length(vortexAxis) > std::numeric_limits<float>::epsilon()) {
		vortexAxis = glm::normalize(vortexAxis);
	}
	const glm::vec3 vortexAcceleration =
		glm::cross(vortexAxis, velocityDirection) *
		local.vorticityMag *
		fieldScale *
		std::max(0.0f, ptsmVorticityGainParam.get());
	const glm::vec3 compressionAcceleration =
		slipDirection *
		local.compression *
		fieldScale *
		std::max(0.0f, ptsmCompressionGainParam.get());
	const glm::vec3 dispersionAcceleration =
		(glm::normalize(slipDirection + velocityDirection * 0.35f)) *
		local.dispersion *
		fieldScale *
		std::max(0.0f, ptsmDispersionGainParam.get());
	const float mass = std::max(ptsmSettings.probe.mass, std::numeric_limits<float>::epsilon());
	glm::vec3 flowForce = desiredAcceleration * mass;
	glm::vec3 tidalForce = (
		tidalAcceleration +
		vortexAcceleration +
		compressionAcceleration +
		dispersionAcceleration
	) * mass;
	glm::vec3 totalForce = flowForce + tidalForce;
	if(ptsmSettings.probe.maxForce > 0.0f) {
		const float forceLength = glm::length(totalForce);
		if(forceLength > ptsmSettings.probe.maxForce && forceLength > std::numeric_limits<float>::epsilon()) {
			const float forceScale = ptsmSettings.probe.maxForce / forceLength;
			flowForce *= forceScale;
			tidalForce *= forceScale;
			totalForce *= forceScale;
		}
	}

	const glm::vec3 previousAcceleration = probe->previousAcceleration;
	const glm::vec3 acceleration = totalForce / mass;
	probe->velocity += std::max(0.0f, dt) * acceleration;
	if(ptsmSettings.probe.maxSpeed > 0.0f) {
		const float speed = glm::length(probe->velocity);
		if(speed > ptsmSettings.probe.maxSpeed && speed > std::numeric_limits<float>::epsilon()) {
			probe->velocity *= ptsmSettings.probe.maxSpeed / speed;
		}
	}
	const float safeDt = std::max(dt, std::numeric_limits<float>::epsilon());
	probe->jerk = glm::length(acceleration - previousAcceleration) / safeDt;
	probe->previousAcceleration = acceleration;
	probe->field.force += totalForce;
	local.flowForce = flowForce;
	local.tidalForce = tidalForce;
	local.totalForce = probe->field.force;
	local.slip = glm::length(localFlowVelocity - probe->velocity);
	ptsmLastGalaxyField = local;
}

glm::mat3 ofApp::outerProduct(const glm::vec3& a, const glm::vec3& b) const {
	glm::mat3 result(0.0f);
	for(int column = 0; column < 3; ++column) {
		for(int row = 0; row < 3; ++row) {
			result[column][row] = a[row] * b[column];
		}
	}
	return result;
}

float ofApp::frobeniusNorm(const glm::mat3& matrix) const {
	float sum = 0.0f;
	for(int column = 0; column < 3; ++column) {
		for(int row = 0; row < 3; ++row) {
			sum += matrix[column][row] * matrix[column][row];
		}
	}
	return std::sqrt(sum);
}

glm::vec3 ofApp::randomUnitVector() const {
	glm::vec3 direction(ofRandomf(), ofRandomf(), ofRandomf());
	const float length = glm::length(direction);
	if(length <= std::numeric_limits<float>::epsilon()) {
		return glm::vec3(1.0f, 0.0f, 0.0f);
	}
	return direction / length;
}

void ofApp::samplePtsmAuditionSpawn(glm::vec3& position, glm::vec3& velocity) const {
	glm::vec3 boundsMin(0.0f);
	glm::vec3 boundsMax(0.0f);
	getVisibleBounds(boundsMin, boundsMax);
	const glm::vec3 boundsSize = glm::max(boundsMax - boundsMin, glm::vec3(std::numeric_limits<float>::epsilon()));
	const glm::vec3 center = 0.5f * (boundsMin + boundsMax);

	auto launchFromLowerRight = [&](const glm::vec3& target) {
		position = glm::vec3(
			boundsMax.x - boundsSize.x * ofRandom(0.06f, 0.16f),
			boundsMin.y + boundsSize.y * ofRandom(0.42f, 0.62f),
			boundsMin.z + boundsSize.z * ofRandom(0.34f, 0.66f)
		);
		const glm::vec3 primary = glm::normalize(target - position);
		glm::vec3 side = glm::cross(primary, glm::vec3(0.0f, 1.0f, 0.0f));
		if(glm::dot(side, side) <= std::numeric_limits<float>::epsilon()) {
			side = glm::cross(primary, glm::vec3(1.0f, 0.0f, 0.0f));
		}
		side = glm::normalize(side);
		const glm::vec3 up = glm::normalize(glm::cross(side, primary));
		const glm::vec3 direction = glm::normalize(
			primary * 0.93f +
			side * ofRandom(-0.12f, 0.18f) +
			up * ofRandom(0.02f, 0.16f)
		);
		const float mass = std::max(ptsmSettings.probe.mass, std::numeric_limits<float>::epsilon());
		const float kinetic = ptsmSettings.injection.maxKineticEnergy * ptsmSettings.injection.energyRatio;
		const float speed = std::sqrt(std::max(0.0f, 2.0f * kinetic / mass));
		velocity = direction * speed;
	};

	static const std::vector<glm::vec3> emptyAttractors;
	const auto& attractors = ptsmField.getFrameCount() > 0 ? ptsmField.getFrame(0) : emptyAttractors;
	if(attractors.size() >= 32) {
		glm::vec3 mean(0.0f);
		for(const auto& point : attractors) {
			mean += point;
		}
		mean /= static_cast<float>(attractors.size());

		glm::vec3 variance(0.0f);
		for(const auto& point : attractors) {
			const glm::vec3 delta = point - mean;
			variance += delta * delta;
		}
		int axis = 0;
		if(variance.y > variance.x && variance.y >= variance.z) {
			axis = 1;
		} else if(variance.z > variance.x && variance.z > variance.y) {
			axis = 2;
		}

		glm::vec3 lowSum(0.0f);
		glm::vec3 highSum(0.0f);
		std::size_t lowCount = 0;
		std::size_t highCount = 0;
		for(const auto& point : attractors) {
			if(point[axis] < mean[axis]) {
				lowSum += point;
				++lowCount;
			} else {
				highSum += point;
				++highCount;
			}
		}

		if(lowCount > 0 && highCount > 0) {
			const glm::vec3 lowCenter = lowSum / static_cast<float>(lowCount);
			const glm::vec3 highCenter = highSum / static_cast<float>(highCount);
			const glm::vec3 lowerRightReference(
				boundsMax.x,
				boundsMin.y,
				center.z
			);
			const float lowDistance = glm::distance2(lowCenter, lowerRightReference);
			const float highDistance = glm::distance2(highCenter, lowerRightReference);
			const glm::vec3 targetCenter = lowDistance < highDistance ? highCenter : lowCenter;
			launchFromLowerRight(targetCenter);
			return;
		}
	}

	launchFromLowerRight(center);
}

void ofApp::choosePtsmSpawn(glm::vec3& position, glm::vec3& velocity) {
	if(ptsmLockedSpawnPosition.has_value() && ptsmLockedSpawnVelocity.has_value()) {
		position = *ptsmLockedSpawnPosition;
		velocity = *ptsmLockedSpawnVelocity;
		return;
	}
	if(!ptsmCurrentSpawnPosition.has_value() || !ptsmCurrentSpawnVelocity.has_value()) {
		samplePtsmAuditionSpawn(position, velocity);
		ptsmCurrentSpawnPosition = position;
		ptsmCurrentSpawnVelocity = velocity;
		return;
	}
	position = *ptsmCurrentSpawnPosition;
	velocity = *ptsmCurrentSpawnVelocity;
}

void ofApp::spawnPtsmProbe() {
	if(!preloadReady || renderFrames.empty()) {
		statusMessage = "PTSM needs loaded frames";
		return;
	}

	syncPtsmSettings();
	rebuildPtsmFieldIfNeeded();
	if(ptsmField.getFrameCount() == 0 || ptsmField.getFrame(0).empty()) {
		statusMessage = "PTSM field has no attractors";
		return;
	}

	glm::vec3 origin(0.0f);
	glm::vec3 velocity(0.0f);
	choosePtsmSpawn(origin, velocity);
	const float speed = glm::length(velocity);
	const glm::vec3 direction = speed > std::numeric_limits<float>::epsilon()
		? velocity / speed
		: randomUnitVector();
	clearPtsmProbes();
	ptsmEngine.spawnProbe(origin, velocity);
	ptsmDisplayPosition = origin;
	ptsmDisplayVelocity = velocity;
	cameraForward = direction;
	const glm::vec3 desiredRight = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), cameraForward);
	if(glm::dot(desiredRight, desiredRight) > std::numeric_limits<float>::epsilon()) {
		cameraRight = glm::normalize(desiredRight);
	}
	cameraStateInitialized = false;
	ptsmTrailCachedSize = 0;
	statusMessage = "Injected PTSM probe " + ofToString(ptsmEngine.getProbeCount());
}

void ofApp::lockPtsmSpawn() {
	if(!ptsmCurrentSpawnPosition.has_value() || !ptsmCurrentSpawnVelocity.has_value()) {
		glm::vec3 position(0.0f);
		glm::vec3 velocity(0.0f);
		samplePtsmAuditionSpawn(position, velocity);
		ptsmCurrentSpawnPosition = position;
		ptsmCurrentSpawnVelocity = velocity;
	}

	ptsmLockedSpawnPosition = ptsmCurrentSpawnPosition;
	ptsmLockedSpawnVelocity = ptsmCurrentSpawnVelocity;
	statusMessage = "Locked PTSM spawn";
	ofLogNotice() << "[collide-vis] Locked PTSM spawn at "
		<< ptsmLockedSpawnPosition->x << ", "
		<< ptsmLockedSpawnPosition->y << ", "
		<< ptsmLockedSpawnPosition->z
		<< " velocity "
		<< ptsmLockedSpawnVelocity->x << ", "
		<< ptsmLockedSpawnVelocity->y << ", "
		<< ptsmLockedSpawnVelocity->z;
}

void ofApp::clearPtsmProbes() {
	ptsmEngine.clearProbes();
	ptsmLastGalaxyField = GalaxyFieldLocal();
	ptsmTrailCache.clear();
	ptsmSmoothedTrailCache.clear();
	ptsmTrailCachedSize = 0;
	statusMessage = "Cleared PTSM probes";
}

void ofApp::resetAllState() {
	playing = false;
	playbackFramePosition = 0.0;
	currentFrameIndex = 0;
	currentFrameBlend = 0.0f;
	rotationDegrees = 0.0f;
	lastRotationCueWeight = 0.0f;

	clearPtsmProbes();
	ptsmCurrentSpawnPosition.reset();
	ptsmCurrentSpawnVelocity.reset();
	ptsmLockedSpawnPosition.reset();
	ptsmLockedSpawnVelocity.reset();
	ptsmDisplayPosition = glm::vec3(0.0f);
	ptsmDisplayVelocity = glm::vec3(0.0f);

	cameraMode = 0;
	cameraForward = glm::vec3(0.0f, 0.0f, -1.0f);
	cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
	smoothedCameraPosition = glm::vec3(0.0f);
	smoothedCameraTarget = glm::vec3(0.0f);
	cameraStateInitialized = false;

	if(preloadReady && !renderFrames.empty()) {
		applyPlaybackPosition(true);
		syncFrameState();
	}
	resetCameraView();
	cameraTimeline.refreshBaseCamera();
	lastUpdateTime = ofGetElapsedTimef();
	statusMessage = "Reset all";
}

void ofApp::updatePtsmDisplayState(double deltaSeconds) {
	const ptsm::ProbeState* probe = ptsmEngine.getProbeState(0);
	if(probe == nullptr) {
		return;
	}

	const float normalizedDelta = static_cast<float>(std::max(0.0, deltaSeconds) * 60.0);
	const float blend = 1.0f - std::pow(
		1.0f - ofClamp(particleDisplaySmoothing, 0.0f, 0.999f),
		normalizedDelta
	);
	ptsmDisplayPosition = glm::mix(ptsmDisplayPosition, probe->position, blend);
	ptsmDisplayVelocity = glm::mix(ptsmDisplayVelocity, probe->velocity, blend);
}

void ofApp::updateCameraFromPtsmProbe() {
	if(cameraMode == 0 || ptsmEngine.getProbeCount() == 0) {
		return;
	}

	const float speedSquared = glm::dot(ptsmDisplayVelocity, ptsmDisplayVelocity);
	if(speedSquared > std::numeric_limits<float>::epsilon()) {
		const glm::vec3 desiredForward = glm::normalize(ptsmDisplayVelocity);
		const float blend = ofClamp(cameraTurnSmoothing, 0.0f, 1.0f);
		cameraForward = glm::normalize(glm::mix(cameraForward, desiredForward, blend));
	}

	const glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
	glm::vec3 desiredRight = glm::cross(worldUp, cameraForward);
	if(glm::dot(desiredRight, desiredRight) <= std::numeric_limits<float>::epsilon()) {
		desiredRight = cameraRight;
	} else {
		desiredRight = glm::normalize(desiredRight);
	}

	const float rightBlend = ofClamp(cameraTurnSmoothing * 0.8f, 0.0f, 1.0f);
	cameraRight = glm::normalize(glm::mix(cameraRight, desiredRight, rightBlend));
	const glm::vec3 up = glm::normalize(glm::cross(cameraForward, cameraRight));

	const glm::vec3 eyePosition = ptsmDisplayPosition + cameraForward * firstPersonEyeOffset;
	const glm::vec3 targetPosition = eyePosition + cameraForward * firstPersonLookAhead;
	if(!cameraStateInitialized) {
		smoothedCameraPosition = eyePosition;
		smoothedCameraTarget = targetPosition;
		cameraStateInitialized = true;
	} else {
		smoothedCameraPosition = glm::mix(
			smoothedCameraPosition,
			eyePosition,
			ofClamp(cameraPositionSmoothing, 0.0f, 1.0f)
		);
		smoothedCameraTarget = glm::mix(
			smoothedCameraTarget,
			targetPosition,
			ofClamp(cameraTargetSmoothing, 0.0f, 1.0f)
		);
	}

	cam.setPosition(smoothedCameraPosition);
	cam.lookAt(smoothedCameraTarget, up);
}

void ofApp::rebuildPtsmTrailCache(const ptsm::ProbeState& probe) {
	ptsmTrailCache.clear();
	ptsmSmoothedTrailCache.clear();
	if(probe.trail.empty()) {
		ptsmTrailCachedSize = 0;
		return;
	}

	const std::size_t stride = std::max<std::size_t>(
		1,
		probe.trail.size() / static_cast<std::size_t>(ptsmMaxTrailDrawPoints)
	);
	std::size_t index = 0;
	for(const auto& point : probe.trail) {
		if(index % stride == 0 || index + 1 == probe.trail.size()) {
			ptsmTrailCache.addVertex(point);
		}
		++index;
	}

	if(ptsmTrailSmoothingSize > 0 &&
	   ptsmTrailCache.size() > static_cast<std::size_t>(ptsmTrailSmoothingSize + 2)) {
		ptsmSmoothedTrailCache = ptsmTrailCache.getSmoothed(ptsmTrailSmoothingSize, 0.5f);
	}
	ptsmTrailCachedSize = probe.trail.size();
}

ptsm::ListenerBasis ofApp::currentListenerBasis() const {
	const auto normalizeOr = [](const glm::vec3& value, const glm::vec3& fallback) {
		const float length = glm::length(value);
		return length > std::numeric_limits<float>::epsilon() ? value / length : fallback;
	};

	ptsm::ListenerBasis basis;
	if(cameraMode != 0) {
		basis.right = normalizeOr(cameraRight, glm::vec3(1.0f, 0.0f, 0.0f));
		basis.forward = normalizeOr(cameraForward, glm::vec3(0.0f, 0.0f, -1.0f));
		basis.up = normalizeOr(glm::cross(basis.forward, basis.right), glm::vec3(0.0f, 1.0f, 0.0f));
		return basis;
	}
	basis.right = normalizeOr(cam.getXAxis(), glm::vec3(1.0f, 0.0f, 0.0f));
	basis.up = normalizeOr(cam.getYAxis(), glm::vec3(0.0f, 1.0f, 0.0f));
	basis.forward = normalizeOr(-cam.getZAxis(), glm::vec3(0.0f, 0.0f, -1.0f));
	return basis;
}

void ofApp::drawPtsmProbes() {
	if(!ptsmGui.enabled || !ptsmGui.showProbe || ptsmEngine.getProbeCount() == 0) {
		return;
	}

	const ptsm::ProbeState* probe = ptsmEngine.getProbeState(0);
	if(probe == nullptr) {
		return;
	}

	ofPushStyle();
	ofNoFill();
	ofEnableBlendMode(OF_BLENDMODE_ALPHA);
	if(probe->trail.size() >= 2) {
		if(ptsmTrailCachedSize != probe->trail.size()) {
			rebuildPtsmTrailCache(*probe);
		}
		const ofPolyline& drawLine = ptsmSmoothedTrailCache.size() >= 2
			? ptsmSmoothedTrailCache
			: ptsmTrailCache;
		ofSetColor(255, 186, 64, static_cast<int>(ptsmTrailAlpha));
		ofSetLineWidth(1.1f);
		drawLine.draw();
	}
	ofDisableBlendMode();

	if(cameraMode == 0) {
		ofFill();
		ofSetColor(255, 238, 170);
		ofPushMatrix();
		ofTranslate(ptsmDisplayPosition);
		ptsmProbeSphere.draw();
		ofPopMatrix();
	}
	ofPopStyle();
}

void ofApp::sendPtsmMetrics() {
	if(!ptsmSettings.osc.enabled || !ptsmOscReady || ptsmEngine.getProbeCount() == 0) {
		return;
	}

	const ptsm::ProbeMetrics metrics = ptsmEngine.getMetrics(currentListenerBasis());
	if(!metrics.valid) {
		return;
	}
	ofxOscBundle bundle = ptsm::makeProbeMetricsOscBundle(ptsmSettings.osc.prefix, metrics);
	if(ptsmLastGalaxyField.valid) {
		const std::string& prefix = ptsmSettings.osc.prefix;
		const ptsm::ListenerBasis basis = currentListenerBasis();
		const auto project = [](const glm::vec3& value, const ptsm::ListenerBasis& listener) {
			return glm::vec3(
				glm::dot(value, listener.right),
				glm::dot(value, listener.up),
				glm::dot(value, listener.forward)
			);
		};
		const glm::vec3 flowBody = project(ptsmLastGalaxyField.flowVelocity, basis);
		const glm::vec3 totalForceBody = project(ptsmLastGalaxyField.totalForce, basis);
		const glm::vec3 vorticityBody = project(ptsmLastGalaxyField.vorticity, basis);

		ptsm::addFloatMessage(bundle, prefix + "/galaxy/flowX", ptsmLastGalaxyField.flowVelocity.x);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/flowY", ptsmLastGalaxyField.flowVelocity.y);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/flowZ", ptsmLastGalaxyField.flowVelocity.z);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/flowBodyX", flowBody.x);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/flowBodyY", flowBody.y);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/flowBodyZ", flowBody.z);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/forceX", ptsmLastGalaxyField.totalForce.x);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/forceY", ptsmLastGalaxyField.totalForce.y);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/forceZ", ptsmLastGalaxyField.totalForce.z);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/forceBodyX", totalForceBody.x);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/forceBodyY", totalForceBody.y);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/forceBodyZ", totalForceBody.z);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/forceMag", glm::length(ptsmLastGalaxyField.totalForce));
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/slip", ptsmLastGalaxyField.slip);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/density", ptsmLastGalaxyField.density);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/dispersion", ptsmLastGalaxyField.dispersion);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/divergence", ptsmLastGalaxyField.divergence);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/compression", ptsmLastGalaxyField.compression);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/vorticity", ptsmLastGalaxyField.vorticityMag);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/vorticityBodyX", vorticityBody.x);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/vorticityBodyY", vorticityBody.y);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/vorticityBodyZ", vorticityBody.z);
		ptsm::addFloatMessage(bundle, prefix + "/galaxy/shear", ptsmLastGalaxyField.shear);
	}
	ptsmOscSender.sendBundle(bundle);
}

std::string ofApp::cameraModeLabel() const {
	return cameraMode == 1 ? "first-person [V]" : "orbit [V]";
}

void ofApp::addRotationCue() {
	RotationCue cue;
	cue.frame = playbackFramePosition;
	cue.degrees = normalizeDegrees(rotationDegrees);
	cue.influenceFrames = rotationCueInfluenceFrames;

	auto insertIt = std::lower_bound(
		rotationCues.begin(),
		rotationCues.end(),
		cue.frame,
		[](const RotationCue& existing, double frame) {
			return existing.frame < frame;
		}
	);
	if(insertIt != rotationCues.end() && std::abs(insertIt->frame - cue.frame) <= 0.0001) {
		*insertIt = cue;
	} else {
		rotationCues.insert(insertIt, cue);
	}

	statusMessage = "Saved rotation cue " + ofToString(cue.degrees, 1) + " deg";
}

void ofApp::removeNearestRotationCue(double threshold) {
	if(rotationCues.empty()) {
		statusMessage = "No rotation cues";
		return;
	}

	double bestDistance = std::max(0.0, threshold);
	int bestIndex = -1;
	for(std::size_t i = 0; i < rotationCues.size(); ++i) {
		const double distance = std::abs(rotationCues[i].frame - playbackFramePosition);
		if(distance <= bestDistance) {
			bestDistance = distance;
			bestIndex = static_cast<int>(i);
		}
	}

	if(bestIndex < 0) {
		statusMessage = "No nearby rotation cue";
		return;
	}

	const float removedDegrees = rotationCues[static_cast<std::size_t>(bestIndex)].degrees;
	rotationCues.erase(rotationCues.begin() + bestIndex);
	statusMessage = "Removed rotation cue " + ofToString(removedDegrees, 1) + " deg";
}

void ofApp::sortRotationCues() {
	std::sort(
		rotationCues.begin(),
		rotationCues.end(),
		[](const RotationCue& a, const RotationCue& b) {
			return a.frame < b.frame;
		}
	);
}

void ofApp::applyRotationCues(double playbackPosition) {
	lastRotationCueWeight = 0.0f;
	if(!rotationCueEnabled || rotationCues.empty()) {
		rotationDegrees = normalizeDegrees(rotationDegrees);
		return;
	}

	const RotationCue* strongestCue = nullptr;
	float strongestWeight = 0.0f;
	for(const auto& cue : rotationCues) {
		const float influence = std::max(cue.influenceFrames, 0.001f);
		const float normalizedDistance = static_cast<float>(std::abs(playbackPosition - cue.frame)) / influence;
		if(normalizedDistance >= 1.0f) {
			continue;
		}

		float weight = 1.0f - normalizedDistance;
		weight = smoothstep(ofClamp(weight, 0.0f, 1.0f));
		if(weight > strongestWeight) {
			strongestWeight = weight;
			strongestCue = &cue;
		}
	}

	if(strongestCue == nullptr) {
		rotationDegrees = normalizeDegrees(rotationDegrees);
		return;
	}

	lastRotationCueWeight = strongestWeight;
	rotationDegrees = mixDegrees(rotationDegrees, strongestCue->degrees, strongestWeight);
}

void ofApp::adjustRotationDegrees(float deltaDegrees) {
	rotationDegrees = normalizeDegrees(rotationDegrees + deltaDegrees);
	statusMessage = "Rotation angle " + ofToString(rotationDegrees, 1) + " deg";
}

bool ofApp::loadRotationCues(const std::string& path) {
	const std::string resolvedPath = ofToDataPath(path, true);
	if(!ofFile::doesFileExist(resolvedPath)) {
		return false;
	}

	ofJson json;
	try {
		json = ofLoadJson(resolvedPath);
	} catch(const std::exception& error) {
		statusMessage = "Failed to load rotation cues: " + std::string(error.what());
		return false;
	}

	if(!json.contains("cues") || !json["cues"].is_array()) {
		statusMessage = "Rotation cue file has no cues";
		return false;
	}

	rotationCueEnabled = json.value("enabled", rotationCueEnabled);
	rotationCueInfluenceFrames = json.value("influenceFrames", rotationCueInfluenceFrames);

	std::vector<RotationCue> loaded;
	for(const auto& entry : json["cues"]) {
		if(!entry.is_object() || !entry.contains("frame") || !entry.contains("degrees")) {
			continue;
		}

		RotationCue cue;
		cue.frame = entry.value("frame", 0.0);
		cue.degrees = normalizeDegrees(entry.value("degrees", 0.0f));
		cue.influenceFrames = entry.value("influenceFrames", rotationCueInfluenceFrames);
		loaded.push_back(cue);
	}

	rotationCues = std::move(loaded);
	sortRotationCues();
	statusMessage = "Loaded " + ofToString(rotationCues.size()) + " rotation cues";
	return true;
}

bool ofApp::saveRotationCues(const std::string& path) const {
	const std::string resolvedPath = ofToDataPath(path, true);
	ofJson json;
	json["version"] = 1;
	json["enabled"] = rotationCueEnabled;
	json["influenceFrames"] = rotationCueInfluenceFrames;
	json["cues"] = ofJson::array();
	for(const auto& cue : rotationCues) {
		ofJson entry;
		entry["frame"] = cue.frame;
		entry["degrees"] = normalizeDegrees(cue.degrees);
		entry["influenceFrames"] = cue.influenceFrames;
		json["cues"].push_back(entry);
	}

	try {
		ofSavePrettyJson(resolvedPath, json);
	} catch(const std::exception&) {
		return false;
	}
	return true;
}

float ofApp::normalizeDegrees(float degrees) {
	float normalized = std::fmod(degrees, 360.0f);
	if(normalized < 0.0f) {
		normalized += 360.0f;
	}
	return normalized;
}

float ofApp::shortestAngleDelta(float fromDegrees, float toDegrees) {
	float delta = normalizeDegrees(toDegrees) - normalizeDegrees(fromDegrees);
	if(delta > 180.0f) {
		delta -= 360.0f;
	} else if(delta < -180.0f) {
		delta += 360.0f;
	}
	return delta;
}

float ofApp::mixDegrees(float fromDegrees, float toDegrees, float amount) {
	const float t = ofClamp(amount, 0.0f, 1.0f);
	return normalizeDegrees(normalizeDegrees(fromDegrees) + shortestAngleDelta(fromDegrees, toDegrees) * t);
}

float ofApp::smoothstep(float amount) {
	const float t = ofClamp(amount, 0.0f, 1.0f);
	return t * t * (3.0f - 2.0f * t);
}

void ofApp::drawPointCloud() {
	if(currentDisplayParticleCount == 0 || !frameTexture.isAllocated()) {
		return;
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	if(depthParticlesParam) {
		glEnable(GL_DEPTH_TEST);
	} else {
		glDisable(GL_DEPTH_TEST);
	}
	glDepthMask(GL_FALSE);

	const float strideForDensity = densityCompensationParam
		? static_cast<float>(std::max<std::size_t>(1, displayStride))
		: 1.0f;
	const float targetDrawCount = static_cast<float>(std::max(1, maxDrawCountParam.get()));
	const float actualDrawCount = static_cast<float>(std::max<std::size_t>(1, currentDisplayParticleCount));
	const float highLoadPointScale = ofClamp(std::sqrt(targetDrawCount / actualDrawCount), 0.62f, 1.0f);
	const float pointDensityScale =
		ofClamp(std::pow(strideForDensity, 0.14f), 1.0f, 1.55f) * highLoadPointScale;
	const float densityGain = ofClamp(std::pow(strideForDensity, 0.70f), 1.0f, 4.5f);

	pointShader.begin();
	pointShader.setUniformTexture("positionTexture", frameTexture, 0);
	pointShader.setUniformTexture(
		"nextPositionTexture",
		nextFrameTexture.isAllocated() ? nextFrameTexture : frameTexture,
		1
	);
	pointShader.setUniform1i("textureWidth", textureWidth);
	pointShader.setUniform1i("displayStride", static_cast<int>(displayStride));
	pointShader.setUniform1f("pointSize", pointSizeParam);
	pointShader.setUniform1f("pointDensityScale", pointDensityScale);
	pointShader.setUniform1f("sceneScale", sceneScaleParam);
	pointShader.setUniform1f("viewScale", coreScaleParam);
	pointShader.setUniform3f(
		"displayNormalizationCenter",
		displayNormalizationReady ? displayNormalizationCenter : glm::vec3(0.0f)
	);
	pointShader.setUniform1f(
		"displayNormalizationScale",
		displayNormalizationReady ? displayNormalizationScale : 1.0f
	);
	pointShader.setUniform1f("frameBlend", currentFrameBlend);
	pointShader.setUniform1f("brightness", brightnessParam);
	pointShader.setUniform1f("densityGain", densityGain);
	pointShader.setUniform1f("gradientMix", gradientParam);
	pointShader.setUniform1f("typeColorMix", typeColorParam);
	pointShader.setUniform1i("fastFullRes", fullResolutionParam && performanceModeParam ? 1 : 0);
	pointShader.setUniform1i("performanceMode", performanceModeParam ? 1 : 0);
	pointShader.setUniform1i("temporalSmooth", temporalSmoothParam ? 1 : 0);
	pointShader.setUniform1i("showGas", showGasParam ? 1 : 0);
	pointShader.setUniform1i("showDarkMatter", showDarkMatterParam ? 1 : 0);
	pointShader.setUniform1i("showDisk", showDiskParam ? 1 : 0);
	pointShader.setUniform1i("showBulge", showBulgeParam ? 1 : 0);
	pointShader.setUniform1i("showStars", showStarsParam ? 1 : 0);
	pointShader.setUniformMatrix4f("modelViewMatrix", ofGetCurrentMatrix(OF_MATRIX_MODELVIEW));
	pointShader.setUniformMatrix4f("projectionMatrix", ofGetCurrentMatrix(OF_MATRIX_PROJECTION));

	drawMesh.draw();

	pointShader.end();
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthFunc(GL_LESS);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void ofApp::drawBounds() const {
	ofPushStyle();
	ofNoFill();
	ofSetColor(255, 44);
	glm::vec3 boundsMin(0.0f);
	glm::vec3 boundsMax(0.0f);
	getVisibleBounds(boundsMin, boundsMax);
	const glm::vec3 worldCenter = 0.5f * (boundsMin + boundsMax);
	const glm::vec3 worldSize = boundsMax - boundsMin;
	ofDrawBox(worldCenter, worldSize.x, worldSize.y, worldSize.z);
	ofPopStyle();
}

void ofApp::drawHud() const {
	ofDisableDepthTest();
	ofSetColor(255);

	std::ostringstream text;
	text << std::fixed << std::setprecision(3);
	if(preloadReady && !renderFrames.empty() && currentFrameIndex < renderFrames.size()) {
		const auto& frame = renderFrames[currentFrameIndex];
		text << "frame " << frame.frameNumber
			<< "  cached " << (currentFrameIndex + 1) << " / " << renderFrames.size()
			<< "  t=" << frame.time
			<< "  draw=" << currentDisplayParticleCount
			<< " / cached=" << currentParticleCount
			<< '\n';
		text << (playing ? "playing" : "paused")
			<< "  blend=" << currentFrameBlend
			<< "  stride=" << displayStride
			<< "  mode=" << (performanceModeParam ? "fast" : "quality")
			<< "  fullRes=" << (fullResolutionParam ? "on" : "off")
			<< "  ram=" << ramFrameCache.size() << "/" << maxRamCachedFrames
			<< "  " << statusMessage;
		text << '\n'
			<< "cameraPath=" << (cameraTimeline.isEnabled() ? "on" : "off")
			<< "  mode=" << cameraTimeline.getModeName()
			<< "  keys=" << cameraTimeline.size()
			<< "  cueWidth=" << cameraTimeline.getCueInfluenceFrames()
			<< "  cueWeight=" << cameraTimeline.getLastCueWeight()
			<< "  " << cameraTimeline.getStatus();
		text << '\n'
			<< "rotationCues=" << (rotationCueEnabled ? "on" : "off")
			<< "  keys=" << rotationCues.size()
			<< "  angle=" << normalizeDegrees(rotationDegrees)
			<< "  cueWidth=" << rotationCueInfluenceFrames
			<< "  cueWeight=" << lastRotationCueWeight;
		text << '\n'
			<< "ptsm=" << (ptsmGui.enabled ? "on" : "off")
			<< "  probes=" << ptsmEngine.getProbeCount()
			<< "  camera=" << cameraModeLabel()
			<< "  attractors=" << (ptsmField.getFrameCount() > 0 ? ptsmField.getFrame(0).size() : 0)
			<< "  flow=" << ptsmFlowSamplesCurrent.size()
			<< "  sigma=" << ptsmSettings.field.sigma
			<< "  gain=" << ptsmSettings.field.fieldGain
			<< "  flowFollow=" << ptsmFlowCouplingParam.get();
		if(ptsmLastGalaxyField.valid) {
			text << '\n'
				<< "galaxyField"
				<< "  slip=" << ptsmLastGalaxyField.slip
				<< "  div=" << ptsmLastGalaxyField.divergence
				<< "  vort=" << ptsmLastGalaxyField.vorticityMag
				<< "  shear=" << ptsmLastGalaxyField.shear
				<< "  disp=" << ptsmLastGalaxyField.dispersion;
		}
	} else {
		text << statusMessage;
	}
	ofDrawBitmapStringHighlight(text.str(), 18.0f, ofGetHeight() - 44.0f);
	ofEnableDepthTest();
}
