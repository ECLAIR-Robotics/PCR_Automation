xarm-build:
	docker build -t xarm-image .

xarm-run:
	docker run -it --rm --privileged -p 127.0.0.1:3000:3000 -v src/:/ws xarm-image