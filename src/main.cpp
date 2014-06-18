#include <cstdlib>

#include "App.h"

int main() {
	App app;

	bool initOK = app.init();
	if (!initOK)
		exit(EXIT_FAILURE);

	app.run();
	return 0;
}