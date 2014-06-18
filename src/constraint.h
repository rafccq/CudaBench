// ======= Constraint header
#ifndef __Constraint_h
#define __Constraint_h

#include <glm/glm.hpp>


using namespace glm;

class Constraint {
private:
	float rest_distance; 

public:
	int indexP1, indexP2;

	Constraint(): indexP1(0), indexP2(0) {
	}

	void setup(Particle* particleArray) {
		Particle& p1 = particleArray[indexP1];
		Particle& p2 = particleArray[indexP2];

		vec3 vec = p1.getPos() - p2.getPos();
		rest_distance = glm::length(vec);
	}

	void set(int p1, int p2, Particle* particleArray) {
		indexP1 = p1;
		indexP2 = p2;
		
		setup(particleArray);
	}

	__device__ __host__ 
	void apply(Particle* particleArray) {
		Particle& p1 = particleArray[indexP1];
		Particle& p2 = particleArray[indexP2];

		vec3 p1_to_p2 = p2.getPos() - p1.getPos();
		float current_distance = glm::length(p1_to_p2); 
		vec3 correctionVector = p1_to_p2*(1 - rest_distance/current_distance); 
		vec3 correctionVectorHalf = correctionVector*0.5f; 
		p1.offsetPos(correctionVectorHalf); 
		p2.offsetPos(-correctionVectorHalf); 
	}

	void apply(Particle* srcArray, Particle* destArray) {
		Particle& p1 = srcArray[indexP1];
		Particle& p2 = srcArray[indexP2];

		vec3 p1_to_p2 = p2.getPos() - p1.getPos();
		float current_distance = glm::length(p1_to_p2); 
		vec3 correctionVector = p1_to_p2*(1 - rest_distance/current_distance); 
		vec3 correctionVectorHalf = correctionVector*0.5f; 

		Particle& destP1 = destArray[indexP1];
		Particle& destP2 = destArray[indexP2];

		destP1.offsetPos(correctionVectorHalf); 
		destP2.offsetPos(-correctionVectorHalf); 
	}

};

#endif