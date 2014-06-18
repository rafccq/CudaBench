#pragma once

//#include "Mesh.h"

#include "particle.h"
#include "constraint.h"
#include "Cloth.h"

class Camera;

class ClothSim {
	Cloth clothObj;

public:
	ClothSim(): clothObj(14, 10, 64, 64) {
	}
	
	void init();
	void update(float dt, bool useGPU);
	void render(Camera* camera);
};
