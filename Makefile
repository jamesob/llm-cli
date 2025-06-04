
build: 
	go build -o llm ./main.go

install: build
	cp llm $(HOME)/.local/bin
