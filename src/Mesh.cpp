#pragma once

//#include "Mesh.h"
#include <vector>
#include <biggle.h>
#include <GL/glfw.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Mesh.h"
#include "Utils.h"

using namespace std;

Mesh::Mesh() {}

Mesh::~Mesh() {}

void Mesh::genBuffers() {
	// Generate an ID for the vertex buffer.
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);

	glGenBuffers(1, &vertexBufferID);
	glGenBuffers(1, &indexBufferID);

	return;
}

void Mesh::uploadIndexData(unsigned int* indices, int count) {
	glBindVertexArray(vertexArrayID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(unsigned int), indices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	nIndices = count;
}

void Mesh::render() {
	//glEnableVertexAttribArray(1);
	glBindVertexArray(vertexArrayID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);

	glDrawElements(GL_TRIANGLES, nIndices, GL_UNSIGNED_INT, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Mesh::deleteBuffers() {
	// Release the vertex buffer.
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &vertexBufferID);

	// Release the index buffer.
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &indexBufferID);

	// Release the vertex array object.
	glBindVertexArray(0);
	glDeleteVertexArrays(1, &vertexArrayID);
	
	return;
}

