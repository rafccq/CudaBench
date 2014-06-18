#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Utils.h"
#include "App.h"
#include "FlockingSim.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/swizzle.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace glm;

void FlockingSim::init() {
	initMesh();

	float r = 10.0f;
	for (int i = 0; i < N_BOIDS; i++) {
		c_positions[i] = glm::sphericalRand(r);
		//c_positions[i].y += r;
		c_positions[i].y = r*0.5f;
		c_forces[i] = glm::sphericalRand(20.0f);
		c_directions[i] = normalize(glm::sphericalRand(20.0f));
		//c_directions[i] = vec3(1,0,0);
		mColors[i] = vec4(1,1,1,1);
		double angle = rand() * 2 * M_PI;
		c_target[i] = vec3(cos(angle), c_positions[i].y, sin(angle));
	}

	mColors[0] = vec4(1,1,0,1);
	c_positions[0] = vec3(r*0.5f, r*0.5f, 0);
	c_wanderRadius = 6.0f;
	c_sphereDist = 4.0f;
	c_randDisplacement = 1.5f;
}

void FlockingSim::initMesh() {
	mBoidMesh.genBuffers();
	
	const int nv = 5;
	VertexPosColor vertices [nv] = {
		+0.5f, +0.5f, -0.5f, DARKGREY_COLOR,
		+0.5f, -0.5f, -0.5f, DARKGREY_COLOR,
		-0.5f, -0.5f, -0.5f, DARKGREY_COLOR,
		-0.5f, +0.5f, -0.5f, DARKGREY_COLOR,
		+0.0f, +0.0f, +1.0f, GREY_COLOR
	};

	mBoidMesh.uploadVertexData<VertexPosColor>(vertices, nv);

	const int ni = 18;
	unsigned int indices[ni] = {
		1, 3, 0,
		2, 3, 1,

		1, 0, 4,
		0, 3, 4,
		3, 2, 4,
		2, 1, 4
	};

	mBoidMesh.uploadIndexData(indices, ni);
}

void FlockingSim::render(ProgramData* program, const glm::mat4& camMatrix) {
	for (int i = 0; i < N_BOIDS; i++) {
	//for (int i = 0; i < 1; i++) {
		glm::vec3 look = c_directions[i];
		glm::vec3 right = normalize(cross(vec3(0,1,0), look));
		glm::vec3 up = cross(look, right);
		
		glm::mat3 M_rot;
		M_rot[0] = right;		M_rot[1] = up;		M_rot[2] = look;

		glm::mat4 modelToWorldXF = Utils::ConstructMatrix(c_positions[i], M_rot);

		vec4 color(1,1,1,1);
		if (i == 0) color = vec4(1,1,0,1);
		
		glUniformMatrix4fv(program->worldToCameraMatrixUnif, 1, GL_FALSE, glm::value_ptr(camMatrix));
		glUniformMatrix4fv(program->modelToWorldMatrixUnif, 1, GL_FALSE, glm::value_ptr(modelToWorldXF));
		glUniform4fv(program->tintUnif, 1, glm::value_ptr(mColors[i]));
		mBoidMesh.render();
	}

	// target
	vec3& pos = c_positions[0];
	vec3& dir = c_directions[0];

	vec3 targetLocal = c_target[0] + vec3(0,0,1)*c_sphereDist;

	glm::vec3 look = normalize(dir);
	glm::vec3 right = normalize(cross(vec3(0,1,0), look));
	glm::vec3 up = cross(look, right);
		
	glm::mat4 xform;
	xform[0] = vec4(right, 0);		xform[1] = vec4(up, 0);		xform[2] = vec4(look, 0);
	xform[3] = vec4(pos, 1);

	vec4 targetWorld = xform*vec4(targetLocal,1);
	vec3 target = swizzle(targetWorld, X, Y, Z);
	vec4 color(1,0,0,1);
	float s = 0.25f;
	//vec3 target = c_positions[0] + c_target + c_directions[0]*c_sphereDist;
	glm::mat4 modelToWorldXF = Utils::ConstructMatrix(target, mat3());
	glm::mat4 Scale = glm::scale(glm::mat4(1.0f), glm::vec3(s, s, s));
	modelToWorldXF = modelToWorldXF * Scale;
	glUniformMatrix4fv(program->worldToCameraMatrixUnif, 1, GL_FALSE, glm::value_ptr(camMatrix));
	glUniformMatrix4fv(program->modelToWorldMatrixUnif, 1, GL_FALSE, glm::value_ptr(modelToWorldXF));
	glUniform4fv(program->tintUnif, 1, glm::value_ptr(color));
	mBoidMesh.render();
}

void FlockingSim::Serial_update(float dt) {
	for (int i = 0; i < N_BOIDS; i++) {
		calculateForces(i);
		updateBoid(i, dt);
	}
}
void FlockingSim::update(float dt, bool useGPU) {
	if (useGPU) {
		GPU_update(dt);
	}
	else {
		Serial_update(dt);
	}
}

// positions, directions, velocities
__global__ 
void Kernel_CalculateForces(vec3* positions, vec3* directions, vec3* velocities, float dt) {
	int index = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	vec3 pos = positions[index];
	vec3 vel = velocities[index];
	vec3 dir = directions[index];

	vec3 separationForce;
	vec3 alignmentForce;
	vec3 cohesionForce;
	
	vec3 centerOfMass;
	int n_neighbors = 0;

	for (int i = 0; i < N_BOIDS; i++) {
		if (i == index) continue;

		vec3 neighborPos = positions[i];
		vec3 diff = neighborPos - pos;
		float dist = length(diff);

		if (dist < NEIGHBOR_DIST && dist > 1e-7) {
			// separation
			separationForce += normalize(diff)/dist;

			// alignment
			alignmentForce += directions[i];

			// cohesion
			centerOfMass += neighborPos;

			n_neighbors++;
		}
	}

	if (n_neighbors > 0) {
		// alignment
		alignmentForce /= n_neighbors;
		alignmentForce -= dir;

		// cohesion
		centerOfMass /= n_neighbors;
		vec3 toCOM = centerOfMass - pos;
		vec3 targetVel = normalize(toCOM) * MAX_SPEED;
		cohesionForce = targetVel - vel;
	}

	vec3 resultingForce;
	resultingForce += separationForce * 0.7f;
	resultingForce += alignmentForce * 0.4f;
	resultingForce += cohesionForce * 0.1f;

	resultingForce.y = 0;

	// update the boid
	vec3 force = resultingForce;

	vel += force * dt * 2.0f;
	pos += vel * dt;

	float velLenSq = dot(vel, vel);
	if (velLenSq > 1e-7)
		dir = normalize(vel);

	if (velLenSq > MAX_SPEED*MAX_SPEED)
		vel = normalize(vel) * MAX_SPEED;

	// wrap pos
	float LIM_X = 30.0f;
	float LIM_Z = 30.0f;
	if (pos.x > LIM_X) pos.x = -LIM_X;
	if (pos.x < -LIM_X) pos.x = LIM_X;

	if (pos.z > LIM_Z) pos.z = -LIM_Z;
	if (pos.z < -LIM_Z) pos.z = LIM_Z;

	// write back
	positions[index] = pos;
	velocities[index] = vel;
	directions[index] = dir;
}

void FlockingSim::GPU_update(float dt) {
    vec3* dev_positions = 0;
    vec3* dev_directions = 0;
    vec3* dev_velocities = 0;
	//float dt
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
	CHECK_CUDA_ERROR("cudaSetDevice failed.\n");

    // Allocate GPU buffers: positions, directions and velocities
	cudaStatus = cudaMalloc((void**)&dev_positions, N_BOIDS * sizeof(vec3));
	CHECK_CUDA_ERROR("cudaMalloc failed: positions.\n");

	cudaStatus = cudaMalloc((void**)&dev_directions, N_BOIDS * sizeof(vec3));
	CHECK_CUDA_ERROR("cudaMalloc failed: directions.\n");

	cudaStatus = cudaMalloc((void**)&dev_velocities, N_BOIDS * sizeof(vec3));
	CHECK_CUDA_ERROR("cudaMalloc failed: velocities.\n");

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_positions, c_positions, N_BOIDS * sizeof(vec3), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR("cudaMemcpy failed: positions [host -> device]\n");

    cudaStatus = cudaMemcpy(dev_directions, c_directions, N_BOIDS * sizeof(vec3), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR("cudaMemcpy failed: directions [host -> device]\n");

    cudaStatus = cudaMemcpy(dev_velocities, c_velocities, N_BOIDS * sizeof(vec3), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR("cudaMemcpy failed: velocities [host -> device]\n");

	// Execute on the GPU!
    Kernel_CalculateForces<<<N_BLOCKS, BLOCK_SIZE>>>(dev_positions, dev_directions, dev_velocities, dt);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
	CHECK_CUDA_ERROR_1("Kernel_CalculateForces launch failed: %s\n", cudaGetErrorString(cudaStatus));
    
	// Wait until the GPU finishes its job
    cudaStatus = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR_1("cudaDeviceSynchronize returned error code %d after launching Kernel_CalculateForces!\n", cudaStatus);

    // Copy vectors back from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c_positions, dev_positions, N_BOIDS * sizeof(vec3), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR("cudaMemcpy failed: positions [device -> host]\n");

    cudaStatus = cudaMemcpy(c_directions, dev_directions, N_BOIDS * sizeof(vec3), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR("cudaMemcpy failed: directions [device -> host]\n");

    cudaStatus = cudaMemcpy(c_velocities, dev_velocities, N_BOIDS * sizeof(vec3), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR("cudaMemcpy failed: velocities [device -> host]\n");

Error:
    cudaFree(dev_positions);
    cudaFree(dev_directions);
    cudaFree(dev_velocities);
}

void FlockingSim::calculateForces(int index) {
	vec3& pos = c_positions[index];

	vec3 separationForce;
	vec3 alignmentForce;
	vec3 cohesionForce;
	
	vec3 centerOfMass;
	int n_neighbors = 0;

	for (int i = 0; i < N_BOIDS; i++) {
		if (i == index) continue;

		vec3& neighborPos = c_positions[i];
		vec3 diff = neighborPos - pos;
		float dist = length(diff);

		if (dist < NEIGHBOR_DIST && dist > 1e-7) {
			mColors[i] = vec4(0,0,1,1);
			// separation
			separationForce += normalize(diff)/dist;

			// alignment
			alignmentForce += c_directions[i];

			// cohesion
			centerOfMass += neighborPos;

			n_neighbors++;
		}
		else
			mColors[i] = vec4(1,1,1,1);
	}

	if (n_neighbors > 0) {
		// alignment
		alignmentForce /= n_neighbors;
		alignmentForce -= c_directions[index];

		// cohesion
		centerOfMass /= n_neighbors;
		vec3 toCOM = centerOfMass - pos;
		vec3 targetVel = normalize(toCOM) * MAX_SPEED;
		cohesionForce = targetVel - c_velocities[index];
	}

	vec3 resultingForce;
	resultingForce += separationForce * 0.7f;
	resultingForce += alignmentForce * 0.4f;
	resultingForce += cohesionForce * 0.1f;

	resultingForce.y = 0;
	c_forces[index] = resultingForce;
}

void FlockingSim::updateBoid(int index, float dt) {
	vec3& pos	= c_positions[index];
	vec3& vel	= c_velocities[index];
	vec3& force = c_forces[index];
	vec3& dir	= c_directions[index];

	vel += force * dt * 2.0f;
	pos += vel * dt;

	//vel += force * 0.01f;
	//pos += vel * 0.01f;

	float velLenSq = dot(vel, vel);
	if (velLenSq > 1e-7)
		c_directions[index] = normalize(vel);

	if (velLenSq > MAX_SPEED*MAX_SPEED)
		vel = normalize(vel) * MAX_SPEED;

	// wrap pos
	float LIM_X = 30.0f;
	float LIM_Z = 30.0f;
	if (pos.x > LIM_X) pos.x = -LIM_X;
	if (pos.x < -LIM_X) pos.x = LIM_X;

	if (pos.z > LIM_Z) pos.z = -LIM_Z;
	if (pos.z < -LIM_Z) pos.z = LIM_Z;
}

vec3 FlockingSim::wander(int index) {
	//vec3 force;
	vec3& pos = c_positions[index];
	vec3& dir = c_directions[index];

	float rx = Utils::frand_norm(c_randDisplacement);
	float ry = Utils::frand_norm(c_randDisplacement);
	float rz = Utils::frand_norm(c_randDisplacement);
	vec3 randVec(rx, ry, rz);
	//randVec = normalize(randVec);// * c_wanderRadius;

	//c_target -= dir*c_sphereDist;
	c_target[index] += randVec;
	c_target[index] = normalize(c_target[index])*c_wanderRadius;
	//c_target.y = 0;
	//c_target += dir*c_sphereDist;
	//vec3 sphereCenter = pos + dir*c_sphereDist;
	//c_target = sphereCenter + randVec;
	vec3 targetLocal = c_target[index] + vec3(0,0,c_sphereDist);

	glm::vec3 look = dir;
	glm::vec3 right = normalize(cross(vec3(0,1,0), look));
	glm::vec3 up = cross(look, right);
		
	glm::mat4 xform;
	xform[0] = vec4(right, 0);		xform[1] = vec4(up, 0);		xform[2] = vec4(look, 0);
	xform[3] = vec4(pos, 1);

	vec4 targetWorld = xform*vec4(targetLocal,1);
	
	vec3 force = swizzle(targetWorld, X, Y, Z);
	force -= pos;
	force.y = 0;
	return force;
}
