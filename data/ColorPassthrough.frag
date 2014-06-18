#version 330

smooth in vec4 interpColor;
//in vec2 inTexCoord;
uniform vec4 tint;

out vec4 outputColor;

void main() {
	outputColor = tint*interpColor;
}
