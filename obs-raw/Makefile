.PHONY: all run dist clean

all: node_modules

node_modules: package.json
	npm install

run: node_modules
	npm run dev

dist: node_modules
	npm run build

clean:
	rm -rf dist node_modules
