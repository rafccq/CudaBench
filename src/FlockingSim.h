#pragma once

#include "Mesh.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define BLACK_COLOR	0.0f, 0.0f, 0.0f, 0.5f
#define WHITE_COLOR	1.0f, 1.0f, 1.0f, 1.0f
#define GREEN_COLOR	0.0f, 1.0f, 0.0f, 1.0f
#define BLUE_COLOR 	0.0f, 0.0f, 1.0f, 1.0f
#define RED_COLOR	1.0f, 0.0f, 0.0f, 0.5f
#define GREY_COLOR	0.8f, 0.8f, 0.8f, 1.0f
#define DARKGREY_COLOR	0.15f, 0.15f, 0.15f, 1.0f
#define BROWN_COLOR 0.5f, 0.5f, 0.0f, 1.0f

struct ProgramData;

/* ------------------------------------------------------------------------------------------
	CUDA and simlulation parameters
--------------------------------------------------------------------------------------------*/
#define BLOCK_SIZE 32
#define N_BLOCKS 16
#define N_BOIDS BLOCK_SIZE*N_BLOCKS

#define MAX_SPEED 20.0f
#define NEIGHBOR_DIST 10.0f

class FlockingSim {
	Mesh mBoidMesh;
	
	glm::vec3 c_forces[N_BOIDS];
	glm::vec3 c_velocities[N_BOIDS];
	glm::vec3 c_positions[N_BOIDS];
	glm::vec3 c_directions[N_BOIDS];
	glm::vec3 c_target[N_BOIDS];
	glm::vec4 mColors[N_BOIDS];

	float c_speeds[N_BOIDS];

	/* ------------------------------------------------------------------------------------------
		Constants
	--------------------------------------------------------------------------------------------*/
	float c_randDisplacement;
	float c_jitter;
	float c_sphereDist;
	float c_wanderRadius;
	//glm::vec3 c_target;//=c_pos+c_dir*c_sphereDist;


	/* ------------------------------------------------------------------------------------------
		Functions
	--------------------------------------------------------------------------------------------*/
	void updateBoid(int index, float dt);
	glm::vec3 wander(int index);
	void calculateForces(int index);
public:
	void init();
	void initMesh();
	void update(float dt, bool useGPU);
	void GPU_update(float dt);
	void Serial_update(float dt);
	void render(ProgramData* program, const glm::mat4& camMatrix);
};
