xarm-build:
	docker build -t xarm-image .

xarm-run:
	docker run -it --rm --privileged \
	-e DISPLAY=$(shell ifconfig en0 | grep inet | awk '$$1=="inet" {print $$2}'):0 \
	-p 127.0.0.1:3000:3000 \
	-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
	-v $(shell pwd)/src:/ws/src \
	--name xarm-container xarm-image