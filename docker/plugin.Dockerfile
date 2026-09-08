# Sandbox image for untrusted plugin execution (Phase B2).
#
# Build: docker build -f docker/plugin.Dockerfile -t surgeval-plugin:local .
# The digest-pinned reference a task names comes from pushing this image to a
# registry; local tags are for development only and never satisfy the
# RuntimeDescriptor digest requirement.
FROM python:3.13-slim
RUN pip install --no-cache-dir "pydantic>=2.7,<3" "numpy>=1.26" "cloudpickle>=3.0"
COPY src/or_audit /opt/surgeval/or_audit
COPY src/surgeval /opt/surgeval/surgeval
ENV PYTHONPATH=/opt/surgeval
