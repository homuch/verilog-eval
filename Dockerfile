FROM ghcr.io/astral-sh/uv:debian
RUN apt-get update && apt-get install -y git wget util-linux
RUN apt-get install -y autoconf gperf make gcc g++ bison flex

WORKDIR /root
RUN wget https://github.com/steveicarus/iverilog/archive/refs/tags/v12_0.zip \
    && unzip v12_0.zip && cd iverilog-12_0 \
    && sh ./autoconf.sh && ./configure && make -j4\
    && make install

WORKDIR /app
COPY pyproject.toml uv.lock .python-version /app/

RUN uv sync
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

RUN mkdir -p /app/build

WORKDIR /app/build

CMD ["../configure", "&&", "make"]

# ../configure \
#     --with-task=$task \
#     --with-model=$model \
#     --with-examples=$shots \
#     --with-samples=$samples \
#     --with-temperature=$temperature \
#     --with-top-p=$top_p \
