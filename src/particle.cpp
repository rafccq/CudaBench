#include "particle.h"

using namespace glm;

void Particle::addForce(vec3 f) {
	acceleration += f/mass;
}

void Particle::timeStep() {
	if(movable) {
		vec3 temp = pos;
		pos = pos + (pos-old_pos)*(1.0f-DAMPING) + acceleration*TIME_STEPSIZE2;
		old_pos = temp;
		acceleration = vec3(0,0,0); 
	}
}

//__device__ __host__
//vec3& Particle::getPos() {return pos;}

void Particle::resetAcceleration() {acceleration = vec3(0,0,0);}

//__device__ __host__
//void Particle::offsetPos(const vec3 v) { if(movable) pos += v;}

void Particle::makeUnmovable() {movable = false;}

void Particle::addToNormal(vec3 normal) {
	if (glm::length(normal) > 0) {
		accumulated_normal += glm::normalize(normal);
	}
}

vec3& Particle::getNormal() { return accumulated_normal;} // notice, the normal is not unit length

void Particle::resetNormal() {accumulated_normal = vec3(0,0,0);}