
#pragma once

// OpenGL
#include <Biggle.h>
#include "Utils.h"

// GLFW
#include <GL/glfw.h>
#include <GL/glut.h> 

//#include <cstdint>

// GLM libraries - Math
// GLM libraries
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/half_float.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/random.hpp>

//#include "tbb/parallel_for.h"
//#include "tbb/blocked_range.h"
#include "App.h"
#include "Input.h"
#include "FlockingSim.h"


extern int keyState[N_KEYS];
extern glm::vec2 mousePos;
extern int mouseButton[3];

using namespace std;
using namespace glm;
static const size_t N = 23;


//#define _OP_ +
//#define OPERATION c[i] = glm::dot(a[i], b[i])
#define OPERATION c[i] = glm::cross(a[i], b[i])

bool initGL() {
	biggleInit();

	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);
	//glFrontFace(GL_CCW);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LEQUAL);
	glDepthRange(0.0f, 1.0f);

	return Utils::checkError("initGL()");
}

void setupCallbacks() {
	glfwSetWindowSizeCallback(window_size_callback);
	glfwSetWindowCloseCallback(window_close_callback);
	glfwSetWindowRefreshCallback(window_refresh_callback);
	glfwSetMouseButtonCallback(mouse_button_callback);
	glfwSetMousePosCallback(mouse_position_callback);
	glfwSetMouseWheelCallback(mouse_wheel_callback);
	//glfwSetCharCallback(char_callback);
	glfwSetKeyCallback(key_callback);

    glfwEnable(GLFW_KEY_REPEAT);
	glfwSwapInterval(1);
}

void initPlaneMesh(Mesh& mesh) {
	mesh.genBuffers();
	
	float sc = 20.0f;
	const int nv = 4;
	VertexPosTex vertices [nv] = {
		+0.5f, 0.0f, +0.5f, +sc,	+0.0f,
		+0.5f, 0.0f, -0.5f, +sc,	+sc,
		-0.5f, 0.0f, -0.5f, +0.0f,  +sc,
		-0.5f, 0.0f, +0.5f, +0.0f,  +0.0f
	};

	mesh.uploadVertexData<VertexPosTex>(vertices, nv);

	unsigned int indices[] = {
		0, 1, 3,
		1, 2, 3
	};

	const int ni = sizeof(indices) / sizeof(int);

	mesh.uploadIndexData(indices, ni);
}

bool App::init() {
	mLastFrameTime = (float) glfwGetTime();
	running = true;
	SCREEN_WIDTH = 1024;
	SCREEN_HEIGHT = 768;
	
	setlocale(LC_ALL, "");

	Utils::SetLocalDataDir("data\\");
	Utils::SetGlobalDataDir("..\\..\\data\\");

	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		return false;
	}

    //Initialize GLFW
	int w = SCREEN_WIDTH, h = SCREEN_HEIGHT;

	int Rbits, Gbits, Bbits, Abits;
	Rbits = Gbits = Bbits = Abits = 8;

	int Sbits = 0, Dbits = 24;
	int mode = GLFW_WINDOW;

	if (!glfwOpenWindow(w, h, Rbits, Gbits, Bbits, Abits, Dbits, Sbits, mode)) {
		glfwTerminate();

		fprintf(stderr, "Failed to create GLFW window");
		return false;
	}

    //Initialize OpenGL
    if(!initGL())
        return false;

	setupCallbacks();
	glfwSetWindowTitle("CUDA Simulation");

	mFlocking.init();
	initPlaneMesh(planeMesh);

	program = LoadProgram("PosColor.vert", "ColorPassthrough.frag");
	glUseProgram(program.progID);
	glUniformMatrix4fv(program.cameraToClipMatrixUnif, 1, GL_FALSE, glm::value_ptr(cam.getCamToClipMatrix()));

	Utils::LoadTexture(&planeTex, "texture\\grid.png");
	planeProgram = LoadProgram("PosTex.vert", "TextureColor.frag");
	planeProgram.textureUnif = glGetUniformLocation(planeProgram.progID, "shaderTexture");

	glUseProgram(planeProgram.progID);
	glUniformMatrix4fv(planeProgram.cameraToClipMatrixUnif, 1, GL_FALSE, glm::value_ptr(cam.getCamToClipMatrix()));

	glUseProgram(0);

	mClothSim.init();

	mGraphFrame.init(600);
	mGraphFrame.pos.x = -0.5f;
	mGraphFrame.pos.y = -0.3f;

	cam.position.x = 0;
	cam.position.y = 20;
	cam.position.z = 70;
	cam.pitch(-15.0f);

	int argc = 1;
	char* argv = "prog";
	glutInit(&argc, &argv);
	//glutDisplayFunc(testFunc);

	return true;
}

ProgramData App::LoadProgram(const std::string &strVertexShader, const std::string &strFragmentShader) {
	GLuint shaderList[2];

	shaderList[0] = Utils::LoadShader(GL_VERTEX_SHADER, strVertexShader);
	shaderList[1] = Utils::LoadShader(GL_FRAGMENT_SHADER, strFragmentShader);

	ProgramData data;
	data.progID = Utils::CreateProgram(shaderList, 2);

	Utils::checkError("App::LoadProgram");

	data.modelToWorldMatrixUnif = glGetUniformLocation(data.progID, "modelToWorldMatrix");
	data.worldToCameraMatrixUnif = glGetUniformLocation(data.progID, "worldToCameraMatrix");
	data.cameraToClipMatrixUnif = glGetUniformLocation(data.progID, "cameraToClipMatrix");
	data.tintUnif = glGetUniformLocation(data.progID, "tint");
	//data.baseColorUnif = glGetUniformLocation(data.progID, "baseColor");
		
	return data;
}

vec3 sp;

void App::testFunc() {
/*
	// ======================== TEST ========================
	glShadeModel(GL_SMOOTH);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_COLOR_MATERIAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//float w = SCREEN_WIDTH, h = SCREEN_HEIGHT;
	//glViewport(0, 0, w, h);
	//glMatrixMode(GL_PROJECTION); 
	//glLoadIdentity();
	//gluPerspective (80, w/h, 1.0f, 5000.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	vec3& pos = cam.position;
	vec3& up = cam.up;
	vec3& f = cam.forward;
	vec3 c = pos + f*50.0f;

	//gluLookAt(pos.x, pos.y, pos.z, 0, 0, -50, up.x, up.y, up.z);
	gluLookAt(pos.x, pos.y, pos.z, c.x, c.y, c.z, up.x, up.y, up.z);

	//mat4 M_View = cam.getWorldToCamMatrix();
	//double V[16];
	//for (int i = 0, k = 0; i < 4; i++)
	//	for (int j = 0; j < 4; j++) {
	//		V[k++] = M_View[i][j];
	//	}

	//glLoadMatrixd(V);

	//glPushMatrix();	
	//glTranslatef(sp.x, sp.y, sp.z);
	sp.z = sin(glfwGetTime())*40;
	
	//glLoadIdentity();
	
	//glColor3f(0.9f, 0.1f, 0.1f);
	//glutSolidSphere(20, 100, 100);


	//glTranslatef(-6.5f, 20, 50.0f); 
	//glRotatef(20,0,1,0);
	//clothObj.addForce(Vec3(0,-0.2,0)*TIME_STEPSIZE2); 
	//clothObj.timeStep(); 
	//clothObj.drawShaded();

	//glPopMatrix();
	//glLoadIdentity();

	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();


	//glLoadIdentity();
	//glPushMatrix();
	//glTranslatef(0, 150, 0);
	//glColor3f(0.9f,0.1f,0.1f);
	//glutSolidSphere(400,50,50);
	//glPopMatrix();


	//glBegin(GL_POLYGON);
	//	glColor3f(0.8f,0.1f,1.0f);
	//	glVertex3f(-200.0f, -100.0f, 0.0f);
	//	glVertex3f( 200.0f, -100.0f, 0.0f);
	//	glColor3f(0.4f,0.4f,0.8f);
	//	glVertex3f( 200.0f, 100.0f, 0.0f);
	//	glVertex3f(-200.0f, 100.0f, 0.0f);
	//glEnd();
	*/
}

void App::renderMesh() {
	glm::vec3 p = cam.position;
	//printf("P = (%.3f, %.3f, %.3f)\n", p.x, p.y, p.z);

	//glEnable(GL_BLEND);
	//glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	const glm::mat4& camMatrix = cam.getWorldToCamMatrix();

	float time0 = (float) glfwGetTime();// / 1000.0f;

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glUseProgram(program.progID);

	//mFlocking.update(mLastFrameTime, false);
	//mFlocking.render(&program, camMatrix);

	// plane:
	float sc = 1000;
	glm::vec3 scaleVec(sc, 1, sc);
	
	glm::mat3 r;
	glm::mat4 planeXForm = Utils::ConstructMatrix(glm::vec3(0, 0, 0), r);
	glm::mat4 scale = glm::scale(glm::mat4(1.0f), scaleVec);
	planeXForm = planeXForm * scale;

	glUseProgram(planeProgram.progID);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, planeTex);

	glUniformMatrix4fv(planeProgram.worldToCameraMatrixUnif, 1, GL_FALSE, glm::value_ptr(camMatrix));
	glUniformMatrix4fv(planeProgram.modelToWorldMatrixUnif, 1, GL_FALSE, glm::value_ptr(planeXForm));
	glUniform1i(planeProgram.textureUnif, 0);

	planeMesh.render();

	//cam.position.y = sin(fElapsedTime) * 1.5f;

	glUseProgram(0);

	mClothSim.update(mLastFrameTime, false);
	mClothSim.render(&cam);

	mLastFrameTime = (float) (glfwGetTime() - time0);
	//printf("%.0fms\n", mLastFrameTime*1000.0f);
	//printf("frame time=%.0fms\n", mLastFrameTime*1000.0f);

	//testFunc();
}

glm::vec2 lastMousePos;
void App::readInput() {
	using namespace glm;
	float speed = 0.05f;
	
	float s = 0.1f;
	vec2 d = mousePos - lastMousePos;
	d *= s;

	if (mouseButton[0] == GLFW_PRESS) {
		cam.rotate(-d.x, -d.y);
		//cam.yaw(-d.x);
	}

	glm::vec3 dir;
	//printf("mouse:dx=%.3f, dy=%.3f\n", d.x, d.y);
	//if (keyState['Q'])
		//dir += cam.forward
	//if (keyState['E'])
		//cam.forward.x += speed*0.12f;
	if (keyState['W'])
		dir += cam.forward;
	if (keyState['S'])
		dir -= cam.forward;
	if (keyState['A'] || keyState[GLFW_KEY_LEFT])
		dir -= cam.right;
	if (keyState['D'] || keyState[GLFW_KEY_RIGHT])
		dir += cam.right;
	if (keyState[GLFW_KEY_UP])
		dir += cam.up;
	if (keyState[GLFW_KEY_DOWN])
		dir -= cam.up;
	if (keyState[GLFW_KEY_ESC])
		running = false;
	if (keyState[GLFW_KEY_LSHIFT])
		speed *= 10.0;

	if (glm::length(dir) != 0)
		dir = glm::normalize(dir);
	cam.position += dir * speed;

	lastMousePos = mousePos;
}

void App::run() {
	while (running) {
		readInput();

		//glClear(GL_COLOR_BUFFER_BIT);
		glm::vec4 bgColor(64, 153, 255, 255);
		bgColor /= 255;
		//glClearBufferfv(GL_COLOR, 0, &bgColor[0]);
		glClearColor(bgColor[0], bgColor[1], bgColor[2], 1.0f);
		glClearDepth(1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		float inv_ratio = (float)SCREEN_HEIGHT / SCREEN_WIDTH;

		// Compute the VP (View Projection matrix)
		float dx = 0, dy = 0;
		glm::mat4 Proj = glm::ortho(-0.5f, 0.5f, -0.5f*inv_ratio, 0.5f*inv_ratio, -1.0f, 1.0f);
		glm::mat4 View = glm::translate(glm::mat4(1.0f), glm::vec3());

		float w = 2.0f;
		float h = w*inv_ratio;
		glm::mat4 Scale = glm::scale(glm::mat4(1.0f), glm::vec3(w, h, 1.0f));
		glm::mat4 ViewProj = Proj * View;//*Scale;
		//glm::mat4 Proj = glm::perspective(45.0f, inv_ratio, 1.0f, 1000.0f);
		glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

		renderMesh();

		// Render graph frame:
		glm::vec3 camPos;
		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);
		View = glm::mat4(1.0f);
		ViewProj = Proj * View*Scale;
		//mGraphFrame.render(Proj, camPos);

		{
			float sc = 0.3f/2.0f;
			float t = (float) glfwGetTime();
			float s1 = sin(t*2.0f)*1.2732f;
			float s2 = sin(t*6.0f)*0.4244f;
			float s3 = sin(t*10.0f)*0.25465f;
			float s4 = sin(t*14.0f)*0.18189f;
			float s5 = sin(t*18.0f)*0.14147f;
		
			s1 = sin(t*1.0f)*0.01f;
			s2 = sin(t*2.0f)*0.002f;
			s3 = sin(t*4.0f)*0.004f;
			//s4 = sin(t*8.0f*s3)*0.08f;
			s4 = sin(t*3.0f)*0.08f;
			//s4 = sin(sin(t*3.0f)*9.0f)*0.08f;
			s5 = cos(t*16.0f+t)*0.032f;
		
			float s = s1+s2+s3+s4+s5;
			//t = t - 10;
			//s = 4.0f*t / (t*t + 1);
			//s = 8.0f / (t*t + 4);
			//graphFrame.addSample(Utils::noise(s*3.0f));

			static float b = 0.2f;
			if (t > 2.0f) b = 0.18f;
			if (t > 3.0f) b = 0.19f;
			s = b + Utils::frand_norm(0.0025f);
			mGraphFrame.addSample(s);
		}
		
		// Update screen
		glfwSwapBuffers();

		if (glfwGetWindowParam(GLFW_OPENED) != GL_TRUE)
			running = false;
	}
}