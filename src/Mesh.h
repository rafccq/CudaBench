#pragma once

#include <vector>
#include <biggle.h>
#include <GL/glfw.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

struct VertexPosColor {
	float position[3];
	float color[4];

	void setupAttribPointers(GLuint vertexBufferID) {
		glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPosColor), (void*)0);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexPosColor), (void*)(3 * sizeof(float)));
	}
};

struct VertexPosTex {
	float position[3];
	float texCoord[2];

	void setupAttribPointers(GLuint vertexBufferID) {
		glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPosTex), (void*)0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(VertexPosTex), (void*)(3 * sizeof(float)));
	}
};

class Mesh {
private:
	GLuint vertexArrayID;
	GLuint vertexBufferID;
	GLuint indexBufferID;

	int nIndices;
public:
	Mesh();
	~Mesh();

	// GPU buffers setup:
	void genBuffers();
	void deleteBuffers();
	
	void render();

	// functions to add data:
	template<typename T>
	void uploadVertexData(T* verticesData, int count) {
		glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
		verticesData[0].setupAttribPointers(vertexBufferID);

		size_t size = sizeof(T);
		glBufferData(GL_ARRAY_BUFFER, count * size, verticesData, GL_STATIC_DRAW);
		glBindVertexArray(0);
	}

	void uploadIndexData(unsigned int* indices, int count);
};