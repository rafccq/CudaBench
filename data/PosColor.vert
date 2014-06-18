#version 330

//layout(location = 0) in vec4 position;
//layout(location = 1) in vec4 color;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord;
//layout(location = 2) in vec3 inNormal;

smooth out vec4 interpColor;
out vec2 texCoord;

uniform mat4 cameraToClipMatrix;
uniform mat4 worldToCameraMatrix;
uniform mat4 modelToWorldMatrix;

void main()
{
	gl_Position = modelToWorldMatrix * vec4(inPosition, 1.0f);
	gl_Position = worldToCameraMatrix * gl_Position;
	gl_Position = cameraToClipMatrix * gl_Position;
	interpColor = inColor;
}