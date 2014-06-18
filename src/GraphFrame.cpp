#include <glm/gtc/matrix_transform.hpp>

#include "GraphFrame.h"
#include "Utils.h"

GraphFrame::GraphFrame() {}

void GraphFrame::init(int numSamples) {
	mFirst = -1;
	mNumSamples = numSamples;

	mVertexData = new Vertex[mNumSamples];

	float dx = (0.5f/mNumSamples);
	for (int i = 0; i < mNumSamples; i++)
		mVertexData[mNumSamples-i-1].pos.x = i * dx;

	initVertexBuffer();
	initVertexArray();
	initProgram();
}

bool GraphFrame::initVertexBuffer() {
	glGenBuffers(1, &mBufferID);

    glBindBuffer(GL_ARRAY_BUFFER, mBufferID);
    glBufferData(GL_ARRAY_BUFFER, mNumSamples*sizeof(Vertex), mVertexData, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	return Utils::checkError("GraphFrame::initVertexBuffer");
}

bool GraphFrame::initVertexArray() {
	glGenVertexArrays(1, &mVertexArrayID);
    glBindVertexArray(mVertexArrayID);
		glBindBuffer(GL_ARRAY_BUFFER, mBufferID);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
		//glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), GLF_BUFFER_OFFSET(sizeof(glm::vec2)));
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glEnableVertexAttribArray(0);
		//glEnableVertexAttribArray(4);
	glBindVertexArray(0);

	return Utils::checkError("GraphFrame::initVertexArray");
}


bool GraphFrame::initProgram() {
	GLuint shaderList[2];

	shaderList[0] = Utils::LoadShader(GL_VERTEX_SHADER, "DefaultVS.vert");
	shaderList[1] = Utils::LoadShader(GL_FRAGMENT_SHADER, "DefaultFS.frag");

	mShaderID = Utils::CreateProgram(shaderList, 2);
	mUniformMVP = glGetUniformLocation(mShaderID, "MVP");

	return Utils::checkError("InitializeProgram");
}

void GraphFrame::addSample(float value) {
	//value = glm::min(value, 0.2f);
	//value = glm::max(value, -0.2f);
	//printf("v=%.2f\n", value);
	if (mFirst < 0) {
		mFirst = 0;
		mVertexData[0].pos.y = value;
	}
	else {
		int start = (mFirst == mNumSamples - 1) ? mNumSamples - 2 : mFirst;
		
		for (int i = start; i >= 0; i--)
			mVertexData[i + 1].pos.y = mVertexData[i].pos.y;
		
		mVertexData[0].pos.y = value;
		mFirst = (mFirst == mNumSamples - 1) ? mFirst : mFirst + 1;
	}

	glBindBuffer(GL_ARRAY_BUFFER, mBufferID);
	glBufferSubData(GL_ARRAY_BUFFER, 0, mNumSamples*sizeof(Vertex), &mVertexData[0]);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

float GraphFrame::getSample(int index) const {
	return mVertexData[index].pos.y;
}

void GraphFrame::render(const glm::mat4 ViewProjMat, glm::vec3 camPos) const {
    //glViewport(0, 0, SCREEN_WIDTH/2, SCREEN_HEIGHT/2);

	glUseProgram(mShaderID);

	// glm::mat4 ViewProjMat
	glm::mat4 Model = glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, pos.y, 0.0f) );
	glm::mat4 MVP = ViewProjMat * Model;

	glUniformMatrix4fv(mUniformMVP, 1, GL_FALSE, &MVP[0][0]);
	glBindVertexArray(mVertexArrayID);
	glDrawArrays(GL_LINE_STRIP, 0, mNumSamples);

	glUseProgram(0);

	Utils::checkError("display");
}