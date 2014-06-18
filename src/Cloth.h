#ifndef __Cloth_h
#define __Cloth_h

#include <vector>
#include <string>
#include "particle.h"
#include "constraint.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glm/glm.hpp>


#define CONSTRAINT_ITERATIONS 10 


class Cloth {
private:

	int num_particles_width; 
	int num_particles_height; 
	
	int num_particles;
	//int num_constraints;


	//std::vector<Particle> particles; 
	//std::vector<Constraint> constraints; 

	Particle* particles;
	//Constraint* constraints;

	Constraint* constraints_1;
	Constraint* constraints_2;
	Constraint* constraints_3;
	Constraint* constraints_4;

	int num_constraints[4];
	
	int numnode,numtri;
	glm::vec3 spherecenter;
	float sphereradius;

	int nConstraints;

	Particle* getParticle(int x, int y);

	void makeConstraint(int block, int x1, int y1, int x2, int y2);

	glm::vec3 calcTriangleNormal(Particle* p1,Particle* p2,Particle* p3);

	void drawTriangle(Particle* p1, Particle* p2, Particle* p3, const glm::vec3 color);

	void Serial_Update();

	void GPU_LaunchKernel(int block, Particle* dev_particles);

	void GPU_updateConstraints();

public:

	Cloth(float width, float height, int num_particles_width, int num_particles_height);

	void drawShaded();

	void timeStep();

	void addForce(const glm::vec3 direction);

	void ballCollision(const glm::vec3 center,const float radius );
};

#endif