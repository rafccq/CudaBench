#include <windows.h> 
#include <stdio.h>
#include <vector>
#include <string>
#include <GL/gl.h>

#include <GL/glfw.h>
#include <glm/glm.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Utils.h"
//#include "particle.h"
#include "Cloth.h"
//#include "constraint.h"


using namespace std;

const int NUM_BLOCKS = 8;
const int THREADS_PER_BLOCK = 4;

inline int even(int n) {
	return n == 0 || (n % 2) == 0;
}

inline int border_R(int b, int w) {
	if (even(w))
		return b == 2 || b == 4;
	
	return b == 1 || b == 3;
}

inline int border_B(int b, int h) {
	if (even(h))
		return b == 3 || b == 4;
	
	return b == 1 || b == 2;
}

inline int get_block(int x, int y) {
	int block_x = 1-even(x/2);
	int block_y = 1-even(y/2);

	int block = 0;
	if (block_x == 0 && block_y == 0) {
		block = 0;
	}
	else if (block_x == 1 && block_y == 1) {
		block = 3;
	}
	else if (block_x == 0 && block_y == 1) {
		block = 2;
	}
	else if (block_x == 1 && block_y == 0) {
		block = 1;
	}

	return block;
}

Cloth::Cloth(float width, float height, int num_particles_width, int num_particles_height) :
	num_particles_width(num_particles_width), 
	num_particles_height(num_particles_height),
	nConstraints(0) {
	
	// setup particles array
	num_particles = num_particles_width*num_particles_height;
	particles = new Particle[num_particles];

	// setup constraints array
	int n_blocks_h = num_particles_width/2;
	int n_odd_h = glm::ceil((float)n_blocks_h*0.5f);
	int n_even_h = n_blocks_h - n_odd_h;

	int n_blocks_v = num_particles_height/2;
	int n_odd_v = glm::ceil((float)n_blocks_v*0.5f);
	int n_even_v = n_blocks_v - n_odd_v;

	int n_blocks_1 = n_odd_h  * n_odd_v;
	int n_blocks_2 = n_even_h * n_odd_v;
	int n_blocks_3 = n_odd_h  * n_even_v;
	int n_blocks_4 = n_even_h * n_even_v;

	int border1[2] = {border_R(1, n_blocks_h), border_B(1, n_blocks_v)};
	int border2[2] = {border_R(2, n_blocks_h), border_B(2, n_blocks_v)};
	int border3[2] = {border_R(3, n_blocks_h), border_B(3, n_blocks_v)};
	int border4[2] = {border_R(4, n_blocks_h), border_B(4, n_blocks_v)};

	int n_constraints1 = 32*n_blocks_1 - 18*border1[0]*n_odd_v	- 18*border1[1]*n_odd_h		+ 10*border1[0]*border1[1];
	int n_constraints2 = 32*n_blocks_2 - 18*border2[0]*n_odd_v	- 18*border2[1]*n_even_h	+ 10*border2[0]*border2[1];
	int n_constraints3 = 32*n_blocks_3 - 18*border3[0]*n_even_v	- 18*border3[1]*n_odd_h		+ 10*border3[0]*border3[1];
	int n_constraints4 = 32*n_blocks_4 - 18*border4[0]*n_even_v	- 18*border4[1]*n_even_h	+ 10*border4[0]*border4[1];

	constraints_1 = new Constraint[n_constraints1];
	constraints_2 = new Constraint[n_constraints2];
	constraints_3 = new Constraint[n_constraints3];
	constraints_4 = new Constraint[n_constraints4];

	num_constraints[0] = num_constraints[1] = num_constraints[2] = num_constraints[3] = 0;

	for (int i = 0; i < 4; i++) {
		int b = i + 1;
		printf("border_R(%d)= %d, border_B(%d)= %d\n", b, border_R(b, n_blocks_h), b, border_B(b, n_blocks_v));
	}

	int nc = num_particles*8 - (num_particles_width+num_particles_height)*9 + 10;
	printf("num_constraints=%d\n", nc);
	//constraints = new Constraint[num_constraints]; //<!>

	for(int x=0; x<num_particles_width; x++) {
		for(int y=0; y<num_particles_height; y++) {
			vec3 pos = vec3(width * (x/(float)num_particles_width),
							-height * (y/(float)num_particles_height),
							0);
			particles[y*num_particles_width+x] = Particle(pos);
		}
	}

	for(int x=0; x<num_particles_width; x++) {
		for(int y=0; y<num_particles_height; y++) {
			int block = get_block(x, y);
			if (x<num_particles_width-1) makeConstraint(block, x, y, x+1, y);
			if (y<num_particles_height-1) makeConstraint(block, x, y, x, y+1);
			if (x<num_particles_width-1 && y<num_particles_height-1) makeConstraint(block, x, y, x+1, y+1);
			if (x<num_particles_width-1 && y<num_particles_height-1) makeConstraint(block, x+1, y, x, y+1);
		}
	}

	for(int x=0; x<num_particles_width; x++) {
		for(int y=0; y<num_particles_height; y++) {
			int block = get_block(x, y);
			if (x<num_particles_width-2) makeConstraint(block, x, y, x+2, y);
			if (y<num_particles_height-2) makeConstraint(block, x, y, x, y+2);
			if (x<num_particles_width-2 && y<num_particles_height-2) makeConstraint(block, x, y, x+2, y+2);
			if (x<num_particles_width-2 && y<num_particles_height-2) makeConstraint(block, x+2, y, x, y+2);
		}
	}

	printf("nc=%d. %d. %d. %d\n", n_constraints1, n_constraints2, n_constraints3, n_constraints4);

	for (int i = 0; i < 4; i++)
		printf("nc_%d=%d\n", i+1, num_constraints[i]);

	//printf("n=%d, nc=%d\n", num_particles, num_constraints);
	for(int i = 0; i < 3; i++) {
		getParticle(0+i ,0)->offsetPos(vec3(0.5,0.0,0.0)); 
		getParticle(0+i ,0)->makeUnmovable(); 

		getParticle(0+i ,0)->offsetPos(vec3(-0.5,0.0,0.0)); 
		getParticle(num_particles_width-1-i ,0)->makeUnmovable();
	}
}

Particle* Cloth::getParticle(int x, int y) {
	return &particles[y*num_particles_width + x];
}

void Cloth::makeConstraint(int block, int x1, int y1, int x2, int y2) {
	int i1 = y1*num_particles_width + x1;
	int i2 = y2*num_particles_width + x2;

	Constraint* constraints_array[4] = {constraints_1, constraints_2, constraints_3, constraints_4};
	Constraint* actual_constraint = constraints_array[block];
	int counter = num_constraints[block];
	actual_constraint[counter++].set(i1, i2, particles);
	num_constraints[block] = counter;
}

vec3 Cloth::calcTriangleNormal(Particle* p1,Particle* p2,Particle* p3) {
	vec3 pos1 = p1->getPos();
	vec3 pos2 = p2->getPos();
	vec3 pos3 = p3->getPos();

	vec3 v1 = pos2-pos1;
	vec3 v2 = pos3-pos1;

	return glm::cross(v1, v2);
}

void Cloth::drawTriangle(Particle *p1, Particle *p2, Particle *p3, const vec3 color) {
	glColor3fv( (GLfloat*) &color );

	vec3 normal1 = p1->getNormal();
	if (glm::length(normal1) > 0) normal1 = glm::normalize(normal1);

	glNormal3f(normal1.x, normal1.y, normal1.z);
	glVertex3fv((GLfloat *) &(p1->getPos() ));
	
	vec3 normal2 = p2->getNormal();
	if (glm::length(normal2) > 0) normal2 = glm::normalize(normal2);

	glNormal3f(normal2.x, normal2.y, normal2.z);
	glVertex3fv((GLfloat *) &(p2->getPos()));

	vec3 normal3 = p3->getNormal();
	if (glm::length(normal3) > 0) normal3 = glm::normalize(normal3);

	glNormal3f(normal3.x, normal3.y, normal3.z);
	glVertex3fv((GLfloat *) &(p3->getPos() ));
}

/*
(x,y)   *--* (x+1,y)
        | /|
        |/ |
(x,y+1) *--* (x+1,y+1)

*/
void Cloth::drawShaded() {
	numnode = 0;
	numtri = 0;

	for(int i = 0; i < num_particles; i++) {
		particles[i].resetNormal();
		numnode += 1;
	}

	for(int x = 0; x<num_particles_width-1; x++) {
		for(int y=0; y<num_particles_height-1; y++) {
			vec3 normal = calcTriangleNormal(getParticle(x+1,y),getParticle(x,y),getParticle(x,y+1));
			getParticle(x+1,y)->addToNormal(normal);
			getParticle(x,y)->addToNormal(normal);
			getParticle(x,y+1)->addToNormal(normal);

			normal = calcTriangleNormal(getParticle(x+1,y+1),getParticle(x+1,y),getParticle(x,y+1));
			getParticle(x+1,y+1)->addToNormal(normal);
			getParticle(x+1,y)->addToNormal(normal);
			getParticle(x,y+1)->addToNormal(normal);
		}
	}

	glBegin(GL_TRIANGLES);
	for(int x = 0; x<num_particles_width-1; x++) {
		for(int y=0; y<num_particles_height-1; y++) {
			vec3 color(0,0,0);
			if (y%3) 
				color = vec3(0.2f,0.47f,0.62f);
			else
				color = vec3(0.9f,0.9f,0.9f);

			drawTriangle(getParticle(x+1,y),getParticle(x,y),getParticle(x,y+1),color);
			numtri += 1;
			drawTriangle(getParticle(x+1,y+1),getParticle(x+1,y),getParticle(x,y+1),color);
			numtri += 1;
		}
	}
	glEnd();
}

//#define CEILING_POS(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
//#define CEILING_NEG(X) ((X-(int)(X)) < 0 ? (int)(X-1) : (int)(X))
//#define CEIL(X) ( ((X) > 0) ? CEILING_POS(X) : CEILING_NEG(X) )

__global__ 
void Kernel_ApplyConstraints(Particle* particles, Constraint* constraints, int total_constraints) {
	int constraints_per_block = 1+total_constraints/NUM_BLOCKS;
	int constraints_per_thread = constraints_per_block/THREADS_PER_BLOCK;

	//constraints_per_block = CEIL(constraints_per_block);
	//constraints_per_thread = CEIL(constraints_per_thread);

	int start = constraints_per_block * blockIdx.x + constraints_per_thread * threadIdx.x;
	int end = start + constraints_per_thread;

	if (end >= total_constraints) end = total_constraints - 1;
	
	for(int i = 0; i < CONSTRAINT_ITERATIONS; i++) {
		for(int j = start; j < end; j++) {
			if (j >= total_constraints)
				break;
			
			constraints[j].apply(particles);
		}
	}
}


void Cloth::Serial_Update() {
	for(int i = 0; i < CONSTRAINT_ITERATIONS; i++) {
		for(int j = 0; j < num_constraints[0]; j++)
			constraints_1[j].apply(particles);

		for(int j = 0; j < num_constraints[2]; j++)
			constraints_3[j].apply(particles);

		for(int j = 0; j < num_constraints[1]; j++)
			constraints_2[j].apply(particles);

		for(int j = 0; j < num_constraints[3]; j++)
			constraints_4[j].apply(particles);
	}
}

void Cloth::timeStep() {
	float t0 = (float) (glfwGetTime());
	
	//Serial_Update();
	GPU_updateConstraints();

	float dt_constraint = (float) (glfwGetTime() - t0);
	
	t0 = (float) (glfwGetTime());
	for(int i = 0; i < num_particles; i++) {
		particles[i].timeStep();
	}

	float dt_particle = (float) (glfwGetTime() - t0);
	
	printf("constraint=%.0fms, particle=%.0fms, c=%d, np=%d\n", dt_constraint*1000.0f, dt_particle*1000.0f, num_constraints, num_particles);
}

void Cloth::GPU_LaunchKernel(int block, Particle* dev_particles) {
	Constraint* constraints_array[4] = {constraints_1, constraints_2, constraints_3, constraints_4};
	int n_constraints = num_constraints[block];

	Constraint* constraints = constraints_array[block];
    //Particle* dev_particles = 0;
    Constraint* dev_constraints = 0;
 //   cudaError_t cudaStatus;

 //   cudaStatus = cudaSetDevice(0);
	//CHECK_CUDA_ERROR("cudaSetDevice failed.\n");

    // Allocate GPU buffers: positions, directions and velocities
	//cudaStatus = cudaMalloc((void**)&dev_particles, num_particles * sizeof(Particle));
	//CHECK_CUDA_ERROR("cudaMalloc failed: directions.\n");

	cudaError_t cudaStatus = cudaMalloc((void**)&dev_constraints, n_constraints * sizeof(Constraint));
	CHECK_CUDA_ERROR("cudaMalloc failed: dev_constraints.\n");

    // Copy input vectors from host memory to GPU buffers.
 //   cudaStatus = cudaMemcpy(dev_particles, particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);
	//CHECK_CUDA_ERROR("cudaMemcpy failed: dev_particles [host -> device]\n");

	cudaStatus = cudaMemcpy(dev_constraints, constraints, n_constraints * sizeof(Constraint), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR("cudaMemcpy failed: dev_constraints [host -> device]\n");

	// Execute on the GPU!
	Kernel_ApplyConstraints<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_particles, dev_constraints, n_constraints);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
	CHECK_CUDA_ERROR_1("Kernel_ApplyConstraints launch failed: %s\n", cudaGetErrorString(cudaStatus));
    
	// Wait until the GPU finishes its job
    cudaStatus = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR_1("cudaDeviceSynchronize returned error code %d after launching Kernel_ApplyConstraints!\n", cudaStatus);

    // Copy vectors back from GPU buffer to host memory: copy only the particles, the constraints don't change
    cudaStatus = cudaMemcpy(particles, dev_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR("cudaMemcpy failed: particles [device -> host]\n");

	//cudaStatus = cudaMemcpy(constraints, dev_constraints, num_constraints * sizeof(Constraint), cudaMemcpyDeviceToHost);
	//CHECK_CUDA_ERROR("cudaMemcpy failed: directions [device -> host]\n");

Error:
    //cudaFree(dev_particles);
    cudaFree(dev_constraints);
}

void Cloth::GPU_updateConstraints() {
	Particle* dev_particles;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
	CHECK_CUDA_ERROR("cudaSetDevice failed.\n");

	cudaStatus = cudaMalloc((void**)&dev_particles, num_particles * sizeof(Particle));
	CHECK_CUDA_ERROR("cudaMalloc failed: dev_particles.\n");

    cudaStatus = cudaMemcpy(dev_particles, particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR("cudaMemcpy failed: dev_particles [host -> device]\n");

	for (int i = 0; i < 4; i++)
		GPU_LaunchKernel(i, dev_particles);

Error:
    cudaFree(dev_particles);
}

void Cloth::addForce(const vec3 direction) {
	for(int i = 0; i < num_particles; i++) {
		particles[i].addForce(direction);
	}

}

void Cloth::ballCollision(const vec3 center,const float radius) {
	for(int i = 0; i < num_particles; i++) {
		Particle& p = particles[i];
		vec3 v = p.getPos()-center;
		float l = glm::length(v);
		if (l < radius) {
			p.offsetPos(glm::normalize(v)*(radius-l)); 
		}
	}

	spherecenter = center;
	sphereradius = radius;
}
