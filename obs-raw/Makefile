.PHONY: all run dist install clean

INSTALL_DIR := A:/obs-raw

all: node_modules

node_modules: package.json
	npm install

run: node_modules
	npm run dev

dist: node_modules
	npm run build

install: dist
	mkdir -p "$(INSTALL_DIR)"
	rm -rf "$(INSTALL_DIR)/assets"
	cp -r dist/assets "$(INSTALL_DIR)/assets"
	cp dist/index.html "$(INSTALL_DIR)/index.html"

clean:
	rm -rf dist node_modules
