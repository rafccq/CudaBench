#version 330

//in vec2 inTexCoord;
in block {
	vec2 Texcoord;
} In;

uniform sampler2D shaderTexture;

out vec4 outputColor;

void main() {
	vec2 uv = In.Texcoord;
	outputColor = texture2D(shaderTexture, uv);
	//outputColor = vec4(1.0f-uv.x);
}
