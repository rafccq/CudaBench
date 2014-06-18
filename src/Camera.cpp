#include <glm/gtc/type_ptr.hpp>

#include "Camera.h"

Camera::Camera(): position(glm::vec3(0.0f)), fov(45.0f) {
	setupCamToClipMatrix(0.001f, 1000.0f);
	init();
}

void Camera::init() {
	right = glm::vec3(1,0,0);
	up = glm::vec3(0,1,0);
	forward = glm::vec3(0,0,-1);
}

Camera::Camera(glm::vec3 _pos, float _fov): position(_pos), fov(_fov) {
	setupCamToClipMatrix(1.0f, 1000.0f);
	init();
}

void Camera::setupWorldToCamMatrix() {
	//glm::vec3 lookDir = glm::normalize(lookPt - position);
	//glm::vec3 upDir = glm::normalize(upPt);

	//glm::vec3 rightDir = glm::normalize(glm::cross(lookDir, upDir));
	//glm::vec3 perpUpDir = glm::cross(rightDir, lookDir);

	//glm::mat4 rotMat;
	//rotMat[0] = glm::vec4(rightDir, 0.0f);
	//rotMat[1] = glm::vec4(perpUpDir, 0.0f);
	//rotMat[2] = glm::vec4(-lookDir, 0.0f);

	//rotMat = glm::transpose(rotMat);

	//glm::mat4 transMat(1.0f);
	//transMat[3] = glm::vec4(-position, 1.0f);

	//worldToCamMatrix = rotMat * transMat;
	// ---------
	//forward = glm::normalize(forward);

	//up = glm::cross(right, forward);
	//up = glm::normalize(up);

	//right = glm::cross(forward, up);
	//right = glm::normalize(right);

	//forward = glm::normalize(forward);

	//glm::mat4 rotMat;
	//rotMat[0] = glm::vec4(right,0);
	//rotMat[1] = glm::vec4(up,0);
	//rotMat[2] = glm::vec4(-forward,0);
	// ----------

	//rotMat = glm::rotate(rotMat, 25.0f, glm::vec3(0, 0, 1));
	//theMat = glm::rotate(theMat, fElapsedTime*50.0f, vec3(0, 1.0F, 0));

	//rotMat = glm::transpose(rotMat);

	//glm::mat4 transMat(1.0f);
	//transMat[3] = glm::vec4(-position, 1.0f);

	//worldToCamMatrix = rotMat * transMat;
	//worldToCamMatrix[0] = rotMat[0];
	//worldToCamMatrix[1] = rotMat[1];
	//worldToCamMatrix[2] = rotMat[2];
	//worldToCamMatrix[3] = glm::vec4(-position, 1.0f);

	// ------------------------------------------

	float x = glm::dot(position, right);
	float y = glm::dot(position, up);
	float z = glm::dot(position, forward);

	worldToCamMatrix[0][0] = right.x;
	worldToCamMatrix[0][1] = up.x;
	worldToCamMatrix[0][2] = -forward.x;
	worldToCamMatrix[0][3] = 0.0f;

	worldToCamMatrix[1][0] = right.y;
	worldToCamMatrix[1][1] = up.y;
	worldToCamMatrix[1][2] = -forward.y;
	worldToCamMatrix[1][3] = 0.0f;

	worldToCamMatrix[2][0] = right.z;
	worldToCamMatrix[2][1] = up.z;
	worldToCamMatrix[2][2] = -forward.z;
	worldToCamMatrix[2][3] = 0.0f;

	worldToCamMatrix[3][0] = -x;
	worldToCamMatrix[3][1] = -y;
	worldToCamMatrix[3][2] = z;
	worldToCamMatrix[3][3] = 1.0f;
}

#define DEG2RAD 3.14159265f/180.0f
void Camera::yaw(float theta) {
	right = right * cos(theta * DEG2RAD) + forward * sin(theta * DEG2RAD);
	right = glm::normalize(right);

	forward = glm::cross(up, right);
	setupWorldToCamMatrix();
}

void Camera::pitch(float theta) {
	forward = forward * cos(theta * DEG2RAD) + up * sin(theta * DEG2RAD);
	forward = glm::normalize(forward);

	up = glm::cross(right, forward);
	setupWorldToCamMatrix();
}

using namespace glm;

void Camera::rotate(float thetaYaw, float thetaPitch) {
	//thetaYaw *= DEG2RAD;
	//thetaPitch *= DEG2RAD;

	vec3 upW(0,1,0);
	quat q = angleAxis(thetaPitch, right);
	//quat q = q0 * angleAxis(thetaPitch, vec3(1,0,0));

	up = q * up;//cross(forward, up);
	up = glm::normalize(up);

	forward = q * forward;
	forward = glm::normalize(forward);

	//float d = dot(right, up);

	q = angleAxis(thetaYaw, upW);
	right = q * right;
	forward = q * forward;
	up = q * up;

	//up = cross(right, forward);
	//up = glm::normalize(up);
	//forward = glm::normalize(forward);
	
	setupWorldToCamMatrix();
}

void Camera::setupCamToClipMatrix(float fzNear, float fzFar) {
	//float fzNear = 1.0f; float fzFar = 45.0f;
	float frustumScale = CalcFrustumScale(fov);

	cameraToClipMatrix[0].x = frustumScale;
	cameraToClipMatrix[1].y = frustumScale;
	cameraToClipMatrix[2].z = (fzFar + fzNear) / (fzNear - fzFar);
	cameraToClipMatrix[2].w = -1.0f;
	cameraToClipMatrix[3].z = (2 * fzFar * fzNear) / (fzNear - fzFar);

	// TODO: extract inverseRatio (hardcoded here as 728.0f/1024.0f)
	//cameraToClipMatrix = glm::perspective(45.0f, 728.0f/1024.0f, 1.0f, 1000.0f);
}


glm::mat4 Camera::getWorldToCamMatrix() {
	setupWorldToCamMatrix();
	return worldToCamMatrix;
}

glm::mat4 Camera::getCamToClipMatrix() {
	return cameraToClipMatrix;
}