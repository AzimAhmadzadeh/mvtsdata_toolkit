# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = mvtsdata_toolkit
SOURCEDIR     = .
BUILDDIR      = _build
TEMPLATES     = _templates
STATIC        = _static

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


# In Makefile two things will happen:
# 1. Three empty directories will be created.
# 2. Catch-all target: route all unknown targets to Sphinx using the new "make mode" option. $(O)
# is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@mkdir -p "$(BUILDDIR)" "$(TEMPLATES)" "$(STATIC)"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	find *.rst -not -name "index.rst" -exec rm -f {} \;
	# find *.rst -not -name "index.rst" -delete
	rm - rf "$(BUILDDIR)" "$(TEMPLATES)" "$(STATIC)"