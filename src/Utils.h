#ifndef _UTILS_H_
#define _UTILS_H_

// include OpenGL
#include <Biggle.h>
#include <string>
#include <vector>
#include <glm/gtx/random.hpp>

#define CHECK_CUDA_ERROR(msg) \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, msg); \
        goto Error; \
	}

#define CHECK_CUDA_ERROR_1(msg, param1) \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, msg, param1); \
        goto Error; \
	}
namespace Utils {
	void SetLocalDataDir(const std::string dir);
	void SetGlobalDataDir(const std::string dir);

	bool checkError(const char* title);
	GLuint CreateShader(GLenum eShaderType, const std::string &strShaderFile, const std::string &strShaderName);
	GLuint LoadShader(GLenum eShaderType, const std::string &strShaderFilename);
	GLuint CreateProgram(const GLuint shaderList[], int n);
	bool LoadTexture(GLuint* texID, const std::string& filename);

	float fract(float n);
	float hash(float n);
	float mix(float a0, float a1, float t);
	float noise(float x);

	inline float frand_norm(float range) {
		float r0 = (float)(rand()) /((float) (RAND_MAX));
		return (r0 * 2 * range - range);
	}

	glm::mat4 ConstructMatrix(glm::vec3 pos, glm::mat3 rot);
}

#endif