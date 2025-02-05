/**
 * @brief Functions for PGM input / output
 *
 * PAE [G4011452] Labs
 * Last update: 17/01/2022
 * Issue date:  01/05/2016
 *
 * Author: Pablo Quesada Barriuso
 *
 * This work is licensed under a Creative Commons
 * Attribution-NonCommercial-ShareAlike 4.0 International.
 * http:// https://creativecommons.org/licenses/by-nc-sa/4.0/
 *
 * THE AUTHOR MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 */

/*
 * Function:  loadPGMu8
 * -----------------------
 * read data from a pgm image file into a matrix x of size wxh as unsigned char
 *
 *  file: input file name
 *
 *  w: width of the image (number of cols)
 *  h: height of the image (number of rows)
 *
 *  returns: a matrix of wxh elements of type unsigned char
 */
unsigned char* loadPGMu8(const char* filename, int* w, int* h)
{
    FILE* fp;
    int i;
    // warning: ignoring return value of ‘int fscanf(FILE*, const char*, ...)’, declared with attribute warn_unused_result
    int f = 0;
    // read buffers
    unsigned char c; char buffer[100];

    fp = fopen (filename, "rb");

    if( fp == NULL ) { fprintf(stderr,"ERROR: Failed to load file: %s\n",filename); return NULL; }

    // clean header of file
    f = fscanf(fp,"%s", buffer); c=fgetc(fp); c=fgetc(fp); if(c!='#')ungetc(c,fp);
    while((c=='#')) { while (c != '\n') { c = fgetc(fp); } }

    // read data dimension from file
    f = fscanf(fp,"%s",buffer); *w=atoi(buffer); f = fscanf(fp,"%s",buffer); *h=atoi(buffer);

    unsigned char* x = NULL;
    x = (unsigned char *) malloc( sizeof(unsigned char) * *w * *h);

    f = fscanf(fp,"%s",buffer); c = fgetc(fp); ;

    int len = (*w)*(*h);

    // read data from file
    for (i = 0; i< len; ++i) { c=fgetc(fp); x[i] = c; }

    fclose(fp);

    printf("[info] Read '%s', %d x %d (%i pixels) (%li Kib)\n", \
        filename, *w, *h, len, (len * sizeof(unsigned char)) / 1024);

    return x;
}
/*
 * Function:  loadPGM32
 * -----------------------
 * read data from a pgm image file into a matrix x of size wxh as float
 *
 *  file: input file name
 *
 *  w: width of the image (number of cols)
 *  h: height of the image (number of rows)
 *
 *  returns: a matrix of wxh elements of type float
 */
float* loadPGM32(const char* filename, int* w, int* h)
{
    FILE* fp;
    int i;
    // warning: ignoring return value of ‘int fscanf(FILE*, const char*, ...)’, declared with attribute warn_unused_result
    int f = 0;
    // read buffers
    unsigned char c; char buffer[100];

    fp = fopen (filename, "rb");

    if( fp == NULL ) { fprintf(stderr,"ERROR: Failed to load file: %s\n",filename); return NULL; }

    // clean header of file
    f = fscanf(fp,"%s", buffer); c=fgetc(fp); c=fgetc(fp); if(c!='#')ungetc(c,fp);
    while((c=='#')) { while (c != '\n') { c = fgetc(fp); } }

    // read data dimension from file
    f = fscanf(fp,"%s",buffer); *w=atoi(buffer); f = fscanf(fp,"%s",buffer); *h=atoi(buffer);

    float* x = NULL;
    x = (float *) malloc( sizeof(float) * *w * *h);

    f = fscanf(fp,"%s",buffer); c = fgetc(fp); ;

    int len = (*w)*(*h);

    // read data from file
    for (i = 0; i< len; ++i) { c=fgetc(fp); x[i] = (float)c; }

    fclose(fp);

    printf("[info] Read '%s', %d x %d (%i pixels) (%li Kib)\n", \
        filename, *w, *h, len, (len * sizeof(float)) / 1024);

    return x;
}

/*
 * Function:  x32Tou8
 * -----------------------
 * normalize data from float to unsigned char
 */
unsigned char* x32Tou8(float *x, int size)
{
    int a = 0, b = 255, i, k;
    unsigned char *xu8 = (unsigned char *)malloc(size*sizeof(unsigned char));
    // initialize minimum and maximum values
    float A = x[0], B = x[0];
    // search the minimum and maximum
    for (k = 0; k < size; ++k)
    {
        A = (x[ k ] < A) ? x[ k ] : A; // A
        B = (x[ k ] > B) ? x[ k ] : B; // B
    }
    for (k = 0; k < size; ++k)
        xu8[ k ] = (unsigned char)floor( ( a+(x[k]-A)*(b-a) / (B-A) ) + 0.5f );

    return xu8;
}

/*
 * Function:  savePGM
 * -----------------------
 * save data to a pgm image file from a matrix x of size wxh
 */
int savePGMu8(const char *filename, unsigned char* x, int w, int h)
{
	FILE *file;
	// read buffer
	unsigned char c;
	int i;

	file = fopen((char*) filename, "w");

	if (file == NULL) { perror("Saving PGM: Can not write to file "); exit(1); }
    // write header to file
    fprintf(file, "P5\n# CREATOR: PAE\n%d %d\n255\n", w, h);

    // write data to file
    for(i = 0; i < w*h; ++i) { c = x[i]; fputc(c,file); }

    fclose(file);

    printf("[info] Saved '%s', %d x %d (%i pixels) (%li Kib)\n", \
        filename, w, h, w*h, (w*h * sizeof(unsigned char)) / 1024);

    return 0;
}

/*
 * Function:  savePGM
 * -----------------------
 * save data to a pgm image file from a matrix x of size wxh
 */
int savePGM32(const char *filename, float* x, int w, int h)
{
	FILE *file;
	// read buffer
	unsigned char c;
	int i;

    // normalize data from float to unsigned char
    unsigned char *xu8 = x32Tou8(x, w*h);

	file = fopen((char*) filename, "w");

	if (file == NULL) { perror("Saving PGM: Can not write to file "); exit(1); }
    // write header to file
    fprintf(file, "P5\n# CREATOR: PAE\n%d %d\n255\n", w, h);

    // write data to file
    for(i = 0; i < w*h; ++i) { c = xu8[i]; fputc(c,file); }

    fclose(file);

    printf("[info] Saved '%s', %d x %d (%i pixels) (%li Kib)\n", \
        filename, w, h, w*h, (w*h * sizeof(unsigned char)) / 1024);

    free(xu8);

    return 0;
}
