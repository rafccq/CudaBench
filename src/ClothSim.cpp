#include <Biggle.h>

#include "ClothSim.h"
#include "Camera.h"

#include <glm/glm.hpp>
#include <GL/glfw.h>
#include <GL/glut.h> 

using namespace glm;

vec3 ballPos(7,-5,0); 
float ballRadius = 2; 


void ClothSim::init() {
	glShadeModel(GL_SMOOTH);
	glClearColor(0.2f, 0.2f, 0.4f, 0.5f);				
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_COLOR_MATERIAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	GLfloat lightPos[4] = {-1.0,1.0,0.5,0.0};
	glLightfv(GL_LIGHT0,GL_POSITION,(GLfloat *) &lightPos);

	glEnable(GL_LIGHT1);

	GLfloat lightAmbient1[4] = {0.0,0.0,0.0,0.0};
	GLfloat lightPos1[4] = {1.0,0.0,-0.2,0.0};
	GLfloat lightDiffuse1[4] = {0.5,0.5,0.5,0.0};

	glLightfv(GL_LIGHT1,GL_POSITION,(GLfloat *) &lightPos1);
	glLightfv(GL_LIGHT1,GL_AMBIENT,(GLfloat *) &lightAmbient1);
	glLightfv(GL_LIGHT1,GL_DIFFUSE,(GLfloat *) &lightDiffuse1);

	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);
}

void ClothSim::update(float dt, bool useGPU) {
	clothObj.addForce(vec3(0,-0.2,0)*TIME_STEPSIZE2);
	clothObj.timeStep();
	clothObj.ballCollision(ballPos, ballRadius); 
}

void ClothSim::render(Camera* camera) {
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_COLOR_MATERIAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	vec3& pos = camera->position;
	vec3& up = camera->up;
	vec3& f = camera->forward;
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
	//sp.z = sin(glfwGetTime())*40;

	glTranslatef(-6.5f, 20, 50.0f);
	glRotatef(20,0,1,0);
	clothObj.drawShaded();

	glPushMatrix(); 
	glTranslatef(ballPos.x, ballPos.y, ballPos.z);
	glColor3f(0.9f,0.1f,0.1f);
	glutSolidSphere(ballRadius-0.1,50,50); 
	glPopMatrix();

	ballPos.z = cos(glfwGetTime()*0.75f)*10;
}