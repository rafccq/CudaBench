#pragma once

#include "Input.h"
#include "Mesh.h"
#include "Camera.h"
#include "GraphFrame.h"
#include "FlockingSim.h"
#include "ClothSim.h"

struct ProgramData {
	GLuint progID;
	GLuint modelToWorldMatrixUnif;
	GLuint worldToCameraMatrixUnif;
	GLuint cameraToClipMatrixUnif;
	GLuint textureUnif;
	GLuint tintUnif;
	//GLuint baseColorUnif;
};

class App {
public:
	bool init();
	void run();

protected:
	int SCREEN_WIDTH;
	int SCREEN_HEIGHT;

	bool running;
	float mLastFrameTime;

	Camera cam;

	ProgramData program;
	Mesh mesh;
	
	Mesh planeMesh;
	GLuint planeTex;
	ProgramData planeProgram;
	GraphFrame mGraphFrame;

	FlockingSim mFlocking;
	ClothSim mClothSim;

	ProgramData LoadProgram(const std::string &strVertexShader, const std::string &strFragmentShader);
	void renderMesh();
	void readInput();

	void testFunc();
};

