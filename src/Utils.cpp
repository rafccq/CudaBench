#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include "Utils.h"
#include "lodepng.h"

namespace Utils {
	// ________ variables
	std::string gLocalDataDir;
	std::string gGlobalDataDir;

	void SetLocalDataDir(const std::string dir) {
		gLocalDataDir = dir;
	}

	void SetGlobalDataDir(const std::string dir) {
		gGlobalDataDir = dir;
	}

	bool checkError(const char* title) {
		int err;
		if((err = glGetError()) != GL_NO_ERROR)
		{
			std::string errorStr;
			switch(err)
			{
			case GL_INVALID_ENUM:
				errorStr = "GL_INVALID_ENUM";
				break;
			case GL_INVALID_VALUE:
				errorStr = "GL_INVALID_VALUE";
				break;
			case GL_INVALID_OPERATION:
				errorStr = "GL_INVALID_OPERATION";
				break;
			case GL_INVALID_FRAMEBUFFER_OPERATION:
				errorStr = "GL_INVALID_FRAMEBUFFER_OPERATION";
				break;
			case GL_OUT_OF_MEMORY:
				errorStr = "GL_OUT_OF_MEMORY";
				break;
			default:
				errorStr = "UNKNOWN";
				break;
			}
			fprintf(stdout, "OpenGL Error(%s): %s\n", errorStr.c_str(), title);
		}

		return err == GL_NO_ERROR;
	}

	const char* GetShaderName(GLenum eShaderType) {
		switch(eShaderType) {
			case GL_VERTEX_SHADER: return "vertex"; break;
			case GL_GEOMETRY_SHADER: return "geometry"; break;
			case GL_FRAGMENT_SHADER: return "fragment"; break;
		}

		return NULL;
	}

	GLuint CreateShader(GLenum eShaderType, const std::string &strShaderFile, const std::string &strShaderName) {
		GLuint shader = glCreateShader(eShaderType);
		const char *strFileData = strShaderFile.c_str();
		glShaderSource(shader, 1, (const GLchar**)&strFileData, NULL);

		glCompileShader(shader);

		GLint status;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
		if (status == GL_FALSE) {
			GLint infoLogLength;
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

			GLchar *strInfoLog = new GLchar[infoLogLength + 1];
			glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

			fprintf(stderr, "Compile failure in %s shader named \"%s\". Error:\n%s\n",
				GetShaderName(eShaderType), strShaderName.c_str(), strInfoLog);
			delete[] strInfoLog;
		}

		return shader;
	}

	GLuint LoadShader(GLenum eShaderType, const std::string &strShaderFilename) {
		std::string strFilename = gLocalDataDir + strShaderFilename;
		std::ifstream shaderFile(strFilename.c_str());
		if(!shaderFile.is_open()) {
			strFilename = gGlobalDataDir + strShaderFilename;
			shaderFile.open(strFilename.c_str());
			if(!shaderFile.is_open()) {
				fprintf(stderr, "Cannot load the shader file \"%s\" for the %s shader.\n",
					strShaderFilename.c_str(), GetShaderName(eShaderType));
				return 0;
			}
		}
		std::stringstream shaderData;
		shaderData << shaderFile.rdbuf();
		shaderFile.close();

		return CreateShader(eShaderType, shaderData.str(), strShaderFilename);
	}

	GLuint CreateProgram(const GLuint shaderList[], int n) {
		GLuint program = glCreateProgram();

		for(int i = 0; i < n; i++)
			glAttachShader(program, shaderList[i]);

		glLinkProgram(program);

		GLint status;
		glGetProgramiv (program, GL_LINK_STATUS, &status);
		if (status == GL_FALSE) {
			GLint infoLogLength;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);

			GLchar *strInfoLog = new GLchar[infoLogLength + 1];
			glGetProgramInfoLog(program, infoLogLength, NULL, strInfoLog);
			fprintf(stderr, "Linker failure: %s\n", strInfoLog);
			delete[] strInfoLog;
		}

		return program;
	}

	bool LoadTexture(GLuint* texID, const std::string& texFilename) {
		std::string strFilename = gLocalDataDir + texFilename;
		std::ifstream texFile(strFilename.c_str());
		if(!texFile.is_open()) {
			strFilename = gGlobalDataDir + texFilename;
			texFile.open(strFilename.c_str());
			if(!texFile.is_open()) {
				fprintf(stderr, "Cannot load the texture file \"%s\n",
					texFilename.c_str());
				return false;
			}
		}

		std::vector<unsigned char> image;
		unsigned width, height;
		unsigned error = lodepng::decode(image, width, height, strFilename, LodePNGColorType::LCT_RGB);

		// Test if there's an error.
		if(error != 0) {
			printf("Texture loading error: %s\n", lodepng_error_text(error));
			return false;
		}

		glGenTextures(1, texID);
		glBindTexture(GL_TEXTURE_2D, *texID);
		
		//int w, h, n;
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,  GL_RGB, GL_UNSIGNED_BYTE, &image[0]);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glGenerateMipmap(GL_TEXTURE_2D);

		GLfloat fLargest;
		glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, fLargest);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

		//printf("load texture: %s, w=%d, h=%d\n", filename, width, height);
	
		std::string msg = std::string("loadTexture:") + strFilename;
		return checkError(msg.c_str());
	}

	float fract(float n) {
		return n - floor(n);
	}

	float hash(float n) {
		return (float) fract(sin(n)*43758.5453123f);
	}

	float mix(float a0, float a1, float t) {
		return a0*(t - 1.0f) + a1*t;
	}

	float noise(float x) {
		float p = floor(x);
		float f = fract(x);

		f = f*f*(3.0f-2.0f*f);

		return mix( hash(p+0.0f), hash(p+1.0f),f);
	}

	glm::mat4 ConstructMatrix(glm::vec3 pos, glm::mat3 rot) {
		glm::mat4 M;

		M[0][0] = rot[0][0];		M[0][1] = rot[0][1];		M[0][2] = rot[0][2];
		M[1][0] = rot[1][0];		M[1][1] = rot[1][1];		M[1][2] = rot[1][2];
		M[2][0] = rot[2][0];		M[2][1] = rot[2][1];		M[2][2] = rot[2][2];

		M[3][0] = pos.x;		M[3][1] = pos.y;	M[3][2] = pos.z;
		return M;
	}
}