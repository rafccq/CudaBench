#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


class Camera {
private:
	glm::mat4 cameraToClipMatrix;
	glm::mat4 worldToCamMatrix;
	float fov;

	// ____ Functions:

	// calculates the frustum scale to build up the cameraToClip matrix
	inline float CalcFrustumScale(float fFovDeg) {
		const float degToRad = 3.14159f * 2.0f / 360.0f;
		float fFovRad = fFovDeg * degToRad;
		return 1.0f / tan(fFovRad / 2.0f);
	}

	void init();
	//void updateMatrix();

public:
	glm::vec3 position;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 forward;

	Camera();
	Camera(glm::vec3 _pos, float _fov);

	void setupCamToClipMatrix(float fzNear, float fzFar);
	void setupWorldToCamMatrix();

	glm::mat4 getWorldToCamMatrix();
	glm::mat4 getCamToClipMatrix();

	void Camera::yaw(float theta);
	void Camera::pitch(float theta);
	void Camera::rotate(float thetaYaw, float thetaPitch);
};
