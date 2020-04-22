FROM kbase/sdkbase2:python
MAINTAINER KBase Developer
# -----------------------------------------
# In this section, you can install any system dependencies required
# to run your App.  For instance, you could place an apt-get update or
# install line here, a git checkout to download code, or run any other
# installation scripts.

# RUN apt-get update

RUN pip install matplotlib==3.2.1 \
    && pip install seaborn==0.10.0 \
    && pip install pandas==1.0.3 \
    && pip install plotly==4.6.0 \
    && pip install scikit-learn==0.22.1 \
    && pip install scipy==1.4.1

# R related installations
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys FCAE2A0E115C3D8A
# RUN echo 'deb https://cloud.r-project.org/bin/linux/debian stretch-cran35/' >> /etc/apt/sources.list

# RUN apt-get update
# RUN apt-get install -y r-base r-base-dev

# RUN cp /usr/bin/R /kb/deployment/bin/.
# RUN cp /usr/bin/Rscript /kb/deployment/bin/.

# vegan: Community Ecology Package
# RUN Rscript -e "install.packages('BiocManager')"
# RUN Rscript -e "BiocManager::install('WGCNA')"
# -----------------------------------------

COPY ./ /kb/module
RUN mkdir -p /kb/module/work
RUN chmod -R a+rw /kb/module

WORKDIR /kb/module

RUN make all

ENTRYPOINT [ "./scripts/entrypoint.sh" ]

CMD [ ]
