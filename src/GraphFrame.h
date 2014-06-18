#ifndef _GRAPH_FRAME_H
#define _GRAPH_FRAME_H

// OpenGL
#include <Biggle.h>
#include <glm/glm.hpp>

struct Vertex {
	Vertex(glm::vec2 const& _pos) :
		pos(_pos)
	{}

	Vertex() :
		pos(glm::vec2(0,0))
	{}

	glm::vec2 pos;
};

class GraphFrame {
	int mFirst;
	int mNumSamples;
	GLuint mVertexArrayID;
	GLuint mBufferID;
	Vertex* mVertexData;
	GLuint mShaderID;
	GLint mUniformMVP;

	bool initVertexBuffer();
	bool initVertexArray();
	bool initProgram();
public:
	glm::vec2 pos;

	GraphFrame();

	void init(int numSamples);
	void addSample(float value);
	void update();
	float getSample(int index) const;
	void render(const glm::mat4 ViewProjMat, glm::vec3 camPos) const;
};

#endif