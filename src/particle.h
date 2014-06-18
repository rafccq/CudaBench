// ======= Particle header
#ifndef __Particle_h
#define __Particle_h

#include <glm/glm.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define DAMPING 0.01f
#define TIME_STEPSIZE2 0.5f*0.5f

class Particle {
private:
	bool movable; 

	float mass; 
	glm::vec3 old_pos; 
	glm::vec3 acceleration; 
	glm::vec3 accumulated_normal; 
	glm::vec3 pos; 

public:

	Particle(glm::vec3 pos) : 
		pos(pos), old_pos(pos), acceleration(glm::vec3(0,0,0)), 
		mass(1), movable(true), accumulated_normal(glm::vec3(0,0,0)){}
	Particle(){}

	void addForce(glm::vec3 f);
	void timeStep();
	
	__device__ __host__ 
	glm::vec3& getPos() {
		return pos;
	}

	void resetAcceleration();

	__device__ __host__ 
	void Particle::offsetPos(const glm::vec3 v) { 
		if(movable) pos += v;
	}

	void makeUnmovable();
	void addToNormal(glm::vec3 normal);
	glm::vec3& getNormal(); 
	void resetNormal();
};

#endif