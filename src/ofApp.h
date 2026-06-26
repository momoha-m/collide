#pragma once

#include "ofMain.h"
#include "ofxGui.h"
#include "ofxCameraTimeline.h"
#include "ofxPtsm.h"
#include "ofxOsc.h"
#include "loader.h"

#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <future>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class ofApp : public ofBaseApp {
public:
	void setup() override;
	void update() override;
	void draw() override;
	void exit() override;

	void keyPressed(int key) override;
	void windowResized(int w, int h) override;

private:
	struct CachedFrameInfo {
		std::size_t sourceIndex = 0;
		int frameNumber = 0;
		double time = 0.0;
		std::size_t particleCount = 0;
		std::array<std::size_t, 6> typeCounts{};
	};

	struct RenderParticle {
		glm::vec3 position = glm::vec3(0.0f);
		glm::vec3 nextPosition = glm::vec3(0.0f);
		float typeIntensity = 0.0f;
	};

	struct PtsmFlowSample {
		glm::vec3 position = glm::vec3(0.0f);
		glm::vec3 velocity = glm::vec3(0.0f);
	};

	struct GalaxyFieldLocal {
		glm::vec3 flowVelocity = glm::vec3(0.0f);
		glm::mat3 gradU = glm::mat3(0.0f);
		glm::vec3 vorticity = glm::vec3(0.0f);
		glm::vec3 flowForce = glm::vec3(0.0f);
		glm::vec3 tidalForce = glm::vec3(0.0f);
		glm::vec3 totalForce = glm::vec3(0.0f);
		float density = 0.0f;
		float dispersion = 0.0f;
		float divergence = 0.0f;
		float compression = 0.0f;
		float shear = 0.0f;
		float slip = 0.0f;
		float vorticityMag = 0.0f;
		bool valid = false;
	};

	struct PrefetchedFrameData {
		std::size_t frameIndex = std::numeric_limits<std::size_t>::max();
		std::vector<RenderParticle> particles;
	};

	struct RotationCue {
		double frame = 0.0;
		float degrees = 0.0f;
		float influenceFrames = 120.0f;
	};

	void setupGui();
	void setupPointShader();
	void setupPtsm();
	std::string resolveDataDirectoryPath() const;

	bool preloadAllFrameData();
	std::filesystem::path buildCacheDirectory() const;
	bool loadPreprocessedCache(const std::filesystem::path& cacheDir);
	bool writePreprocessedCache(const std::filesystem::path& cacheDir) const;
	bool loadCachedFrameData(std::size_t frameIndex, std::vector<RenderParticle>* particles) const;
	bool writeCachedFrameData(std::size_t frameIndex, const std::vector<RenderParticle>& particles) const;
	bool computeDisplayNormalizationFromCache();
	void warmInitialRamCache(std::size_t startFrameIndex);
	void storeRamCachedFrame(std::size_t frameIndex, std::vector<RenderParticle>&& particles);
	const std::vector<RenderParticle>* findRamCachedFrame(std::size_t frameIndex) const;
	std::vector<RenderParticle> buildRenderParticles(
		const SnapshotFrame& frame,
		const glm::vec3& rawCenter,
		float rawScale,
		const std::unordered_map<std::uint64_t, std::size_t>& slotById,
		std::size_t slotCount
	) const;

	bool loadRenderFrame(std::size_t frameIndex);
	bool loadNextRenderFrame(std::size_t frameIndex);
	bool loadAheadRenderFrame(std::size_t frameIndex);
	void uploadCurrentFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles);
	void uploadNextFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles);
	void uploadAheadFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles);
	void populateTextureFromParticles(
		ofTexture& texture,
		ofFloatPixels& pixels,
		std::size_t& uploadedTexelCount,
		const std::vector<RenderParticle>& particles,
		bool clearUnusedTexels
	);
	void rebuildDrawMesh(std::size_t particleCount);
	void syncFrameState();
	void prepareAheadFrameTexture();
	void beginNextFramePrefetch(std::size_t frameIndex);
	void finishNextFramePrefetch();
	void waitForPendingPrefetch();
	void updatePlaybackPosition(double deltaSeconds);
	void applyPlaybackPosition(bool reloadRenderFrame);
	void frameStateForPosition(double playbackPosition, std::size_t& frameIndex, float& frameBlend) const;
	std::size_t getInterpolatedParticleCount() const;
	std::size_t getDisplayParticleCount(std::size_t particleCount) const;
	bool particleTypeEnabled(int type) const;
	float packedTypeIntensity(const SnapshotParticle& particle) const;
	void syncDisplaySettings();
	std::size_t computeDisplayStride(std::size_t renderParticleCount) const;
	void updateInitialCameraFocusFromParticles(const std::vector<RenderParticle>& particles);
	glm::vec3 transformCachedPositionForDisplay(const glm::vec3& position) const;
	glm::vec3 transformCachedPositionForPtsm(const glm::vec3& position) const;
	void getVisibleBounds(glm::vec3& boundsMin, glm::vec3& boundsMax) const;
	void resetCameraView();
	void constrainCameraDistance();
	void syncPtsmSettings();
	void rebuildPtsmFieldIfNeeded();
	bool buildPtsmFrame(std::size_t frameIndex, std::vector<glm::vec3>& attractors) const;
	std::vector<glm::vec3> compressPtsmAttractors(const std::vector<glm::vec3>& points, int maxAttractors) const;
	bool buildPtsmFlowFrame(std::size_t frameIndex, std::vector<PtsmFlowSample>& samples) const;
	std::vector<PtsmFlowSample> compressPtsmFlowSamples(const std::vector<PtsmFlowSample>& samples, int maxSamples) const;
	GalaxyFieldLocal sampleGalaxyField(const glm::vec3& position, const glm::vec3& probeVelocity) const;
	GalaxyFieldLocal sampleGalaxyFieldFromFrame(
		const std::vector<PtsmFlowSample>& samples,
		const glm::vec3& position,
		const glm::vec3& probeVelocity
	) const;
	void applyPtsmFlowForces(float dt);
	glm::mat3 outerProduct(const glm::vec3& a, const glm::vec3& b) const;
	float frobeniusNorm(const glm::mat3& matrix) const;
	glm::vec3 randomUnitVector() const;
	void samplePtsmAuditionSpawn(glm::vec3& position, glm::vec3& velocity) const;
	void choosePtsmSpawn(glm::vec3& position, glm::vec3& velocity);
	void spawnPtsmProbe();
	void lockPtsmSpawn();
	void clearPtsmProbes();
	void resetAllState();
	void updatePtsmDisplayState(double deltaSeconds);
	void updateCameraFromPtsmProbe();
	void rebuildPtsmTrailCache(const ptsm::ProbeState& probe);
	ptsm::ListenerBasis currentListenerBasis() const;
	void drawPtsmProbes();
	void sendPtsmMetrics();
	std::string cameraModeLabel() const;
	void addRotationCue();
	void removeNearestRotationCue(double threshold = 8.0);
	void sortRotationCues();
	void applyRotationCues(double playbackPosition);
	void adjustRotationDegrees(float deltaDegrees);
	bool loadRotationCues(const std::string& path);
	bool saveRotationCues(const std::string& path) const;
	static float normalizeDegrees(float degrees);
	static float shortestAngleDelta(float fromDegrees, float toDegrees);
	static float mixDegrees(float fromDegrees, float toDegrees, float amount);
	static float smoothstep(float amount);
	void drawPointCloud();
	void drawBounds() const;
	void drawHud() const;

	const std::string defaultDataDirectoryPath =
		"/Users/momoha/Library/CloudStorage/Dropbox-SFC-CNS/Momoha Anayama/gadget-4/CollidingGalaxiesSFR";
	std::string dataDirectoryPath;

	GadgetSnapshotLoader loader;
	std::vector<SnapshotFrameInfo> frameInfos;
	std::vector<CachedFrameInfo> renderFrames;
	std::filesystem::path cacheDirectory;

	ofEasyCam cam;
	ofxCameraTimeline cameraTimeline;
	ofVboMesh drawMesh;
	ofShader pointShader;
	ofTexture frameTexture;
	ofTexture nextFrameTexture;
	ofTexture aheadFrameTexture;
	ofFloatPixels frameUploadPixels;
	ofFloatPixels nextFrameUploadPixels;
	ofFloatPixels aheadFrameUploadPixels;
	std::future<PrefetchedFrameData> nextFramePrefetchFuture;
	std::optional<PrefetchedFrameData> stagedNextFrameData;
	std::unordered_map<std::size_t, std::vector<RenderParticle>> ramFrameCache;
	std::deque<std::size_t> ramFrameCacheOrder;
	std::vector<RotationCue> rotationCues;

	ofxPanel gui;
	ofParameterGroup guiParams;
	ofParameter<float> pointSizeParam;
	ofParameter<float> sceneScaleParam;
	ofParameter<float> coreScaleParam;
	ofParameter<float> playbackFpsParam;
	ofParameter<bool> temporalSmoothParam;
	ofParameter<float> brightnessParam;
	ofParameter<float> gradientParam;
	ofParameter<float> typeColorParam;
	ofParameter<int> maxDrawCountParam;
	ofParameter<bool> fullResolutionParam;
	ofParameter<bool> autoLodParam;
	ofParameter<bool> densityCompensationParam;
	ofParameter<bool> depthParticlesParam;
	ofParameter<bool> performanceModeParam;
	ofParameter<bool> showGasParam;
	ofParameter<bool> showDarkMatterParam;
	ofParameter<bool> showDiskParam;
	ofParameter<bool> showBulgeParam;
	ofParameter<bool> showStarsParam;
	ofParameter<bool> showBoundsParam;
	ofParameter<bool> autoRotateParam;
	ofParameter<bool> showGuiParam;
	ofParameter<float> ptsmProbeRadiusParam;
	ofParameter<float> ptsmTrailAlphaParam;
	ofParameter<int> ptsmTrailSmoothingParam;
	ofParameter<float> ptsmDensityMixParam;
	ofParameter<float> ptsmFlowCouplingParam;
	ofParameter<float> ptsmFlowRadiusParam;
	ofParameter<float> ptsmFlowVelocityScaleParam;
	ofParameter<float> ptsmFlowVerticalMixParam;
	ofParameter<float> ptsmTidalGainParam;

	ptsm::Settings ptsmSettings;
	ptsm::GuiBindings ptsmGui;
	ptsm::FieldModel ptsmField;
	ptsm::ProbeParticleEngine ptsmEngine;
	ofxOscSender ptsmOscSender;
	glm::vec3 ptsmDisplayPosition = glm::vec3(0.0f);
	glm::vec3 ptsmDisplayVelocity = glm::vec3(0.0f);
	ofPolyline ptsmTrailCache;
	ofPolyline ptsmSmoothedTrailCache;
	std::size_t ptsmTrailCachedSize = 0;
	ofSpherePrimitive ptsmProbeSphere;
	std::vector<PtsmFlowSample> ptsmFlowSamplesCurrent;
	std::vector<PtsmFlowSample> ptsmFlowSamplesNext;
	GalaxyFieldLocal ptsmLastGalaxyField;

	glm::vec3 worldMin = glm::vec3(-1.35f, -1.05f, -1.05f);
	glm::vec3 worldMax = glm::vec3(1.35f, 1.05f, 1.05f);
	std::size_t currentFrameIndex = 0;
	std::size_t currentParticleCount = 0;
	std::size_t currentDisplayParticleCount = 0;
	std::size_t maxParticleCount = 0;
	std::size_t loadedRenderFrameIndex = std::numeric_limits<std::size_t>::max();
	std::size_t loadedNextRenderFrameIndex = std::numeric_limits<std::size_t>::max();
	std::size_t loadedAheadRenderFrameIndex = std::numeric_limits<std::size_t>::max();
	std::size_t frameUploadedTexelCount = 0;
	std::size_t nextFrameUploadedTexelCount = 0;
	std::size_t aheadFrameUploadedTexelCount = 0;
	std::size_t displayStride = 1;
	int textureWidth = 1;
	int textureHeight = 1;
	bool playing = false;
	bool preloadReady = false;
	double lastUpdateTime = 0.0;
	double playbackFramePosition = 0.0;
	float currentFrameBlend = 0.0f;
	float rotationDegrees = 0.0f;
	float rotationCueInfluenceFrames = 120.0f;
	float lastRotationCueWeight = 0.0f;
	bool rotationCueEnabled = true;
	std::string statusMessage;
	std::size_t ptsmFieldFrameIndex = std::numeric_limits<std::size_t>::max();
	int ptsmLastSampleCount = -1;
	float ptsmLastSceneScale = -1.0f;
	float ptsmLastViewScale = -1.0f;
	bool ptsmOscReady = false;
	std::optional<glm::vec3> ptsmCurrentSpawnPosition;
	std::optional<glm::vec3> ptsmCurrentSpawnVelocity;
	std::optional<glm::vec3> ptsmLockedSpawnPosition;
	std::optional<glm::vec3> ptsmLockedSpawnVelocity;
	int cameraMode = 0;
	glm::vec3 cameraForward = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
	glm::vec3 smoothedCameraPosition = glm::vec3(0.0f);
	glm::vec3 smoothedCameraTarget = glm::vec3(0.0f);
	bool cameraStateInitialized = false;
	float firstPersonLookAhead = 0.16f;
	float firstPersonEyeOffset = 0.012f;
	float cameraTurnSmoothing = 0.14f;
	float particleDisplaySmoothing = 0.32f;
	float cameraPositionSmoothing = 0.2f;
	float cameraTargetSmoothing = 0.24f;
	float ptsmProbeRadius = 0.005f;
	float ptsmTrailAlpha = 140.0f;
	int ptsmTrailSmoothingSize = 8;
	int ptsmMaxTrailDrawPoints = 2500;
	int previousMaxDrawCount = 0;
	bool previousFullResolution = false;
	bool previousAutoLod = true;
	glm::vec3 initialCameraFocus = glm::vec3(0.0f);
	float initialCameraRadius = 1.0f;
	bool initialCameraFocusValid = false;
	glm::vec3 displayNormalizationCenter = glm::vec3(0.0f);
	float displayNormalizationScale = 1.0f;
	bool displayNormalizationReady = false;
	const std::string cameraTimelinePath = "camera_path.json";
	const std::string rotationCuesPath = "rotation_cues.json";

	std::size_t cacheParticleLimit = 2000000;
	std::size_t maxRamCachedFrames = 24;
};
