#version 330

//layout(location = 0) in vec4 position;
//layout(location = 1) in vec4 color;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
//layout(location = 2) in vec3 inNormal;

//out vec2 outTexCoord;

out block {
	vec2 Texcoord;
} Out;

uniform mat4 cameraToClipMatrix;
uniform mat4 worldToCameraMatrix;
uniform mat4 modelToWorldMatrix;

void main()
{
	gl_Position = modelToWorldMatrix * vec4(inPosition, 1.0f);
	gl_Position = worldToCameraMatrix * gl_Position;
	gl_Position = cameraToClipMatrix * gl_Position;
	Out.Texcoord = inTexCoord;
}