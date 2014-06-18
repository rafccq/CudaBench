#version 330 core

#define POSITION	0
#define COLOR		3
#define FRAG_COLOR	0

layout(location = FRAG_COLOR, index = 0) out vec4 Color;

void main() {
	Color = vec4(1.0f, 1.0f, 0.0f, 1.0f);
}