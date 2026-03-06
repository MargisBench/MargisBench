/**
 * This script allows to perfom inferencing specifying a .hef model file, the dataset path (were are sub-directories for each class) and the batch size 
 * on Fusion 844 AI with HailoRT 4.20.1
 * PLEASE NOTE: The batch size is only a value to change the batch-referred metrics; the inferences are always perfomed per single images. 
 **/
#include <hailort.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <time.h>
#include "DatasetFiles/classes.h"

#define MAX_EDGE_LAYERS 16
#define MAX_PATH_LEN 1024
#define DATASET_MAX_IMAGES 5000 // Limit for the file list array

// X-Macro for classes
#define X(id, name, str) str,
static const char *class_names[]= {CLASSES_TABLE};


/** 
* The TimeStats struct allows to store the metrics that will be processed
* and shown at the end of the script about Inference Time and Accuracy. 
**/
typedef struct {
    int batch_size;
    int total;
    int correct;
    int correct_count_class[CLASSES][CLASSES];
    double start_time;
    double end_time;
    double start_time_kernel;
    double end_time_kernel;
    float64_t avg_hw_latency_per_image;
    float64_t avg_hw_latency_per_batch;
    float64_t avg_overall_latency_per_image;
    float64_t avg_overall_latency_per_batch;
    float64_t overall_latency_ms;
} TimeStats;


/** 
 * The MemoryStats struct allows to store the stats that will be processed
 * and shown at the end of the script about Memory Usage. 
 **/
typedef struct {
    double weights;
    double l2_data_usage;
    double boundary_in;
    double boundary_out;
    double inter_in;
    double inter_out;
} MemoryStats;


/**
 * This struct is passed as paremeter to the Input and Output threads. 
 * It's used to feed the Hailo8 accelerator and take the consequential
 * results of the inference, updating TimeStats instance. 
 **/
typedef struct {
    hailo_input_vstream input_vstream;
    hailo_output_vstream output_vstream;
    hailo_configured_network_group network_group;
    
    // File List (Shared by both threads)
    char file_paths[DATASET_MAX_IMAGES][MAX_PATH_LEN];
    int expected_classes[DATASET_MAX_IMAGES]; // To check accuracy
    int total_images;

    // Frame Sizes
    size_t frame_in_size;
    size_t frame_out_size;

    // Stats pointer to write results
    TimeStats *stats;
} ThreadContext;


/** get_time_msec()
 *  Utility function to take the tike through clock_gettime()
 **/
double get_time_msec() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double) (ts.tv_sec * 1000.0) + (double) (ts.tv_nsec / 1e6);
}

/** findMax(uint8_t*) -> int
* 
* This function works as an ArgMax function. 
* Thakes the array of results as input and it'll find the max 
* probability value, that will be assigned to a specific class (Index). 
* This function returns the prediction index (the specific class). 
**/
int findMax(uint8_t* results){
    int prediction = 0;
    uint8_t prediction_value = results[0];
    for (int i = 1; i<CLASSES; i++){ 
        if (results[i] > prediction_value){
            prediction = i; 
            prediction_value = results[i];
        }
    }
    return prediction;
}


/** checkStatus(hailo_status, char*, char*, hailo_hef, hailo_vdevice) -> void
* 
* This function allows to check the Hailo Status returned from some of Hailo Functions. 
* If the status is HAILO_SUCCESS, the esecution will continue, otherwise is stopped. 
* 
**/
void checkStatus(hailo_status status, const char* msg) {
    if (status != HAILO_SUCCESS) {
        printf("[ERROR] %s (Status: %d)\n", msg, status);
        exit(1);
    }
}

/** input_thread_func(void* arg)
 * 
 *  This is the input thread. Reads the images one by one putting it into the buffer and feed the
 *  accelerator through hailo_vstream_write_raw_buffer function.
 **/
void* input_thread_func(void* arg) {
    ThreadContext* ctx = (ThreadContext*)arg;
    hailo_status status;
    
    // Allocate ONE buffer for reading files
    uint8_t* buffer = malloc(ctx->frame_in_size);

    for (int i = 0; i < ctx->total_images; i++) {
        // 1. Read File
        FILE* fd = fopen(ctx->file_paths[i], "rb");
        if (!fd) {
            printf("[WARN] Failed to open %s\n", ctx->file_paths[i]);
            // Fill buffer with zeros to keep synchronization
            memset(buffer, 0, ctx->frame_in_size); 
        } else {
            fread(buffer, ctx->frame_in_size, 1, fd);
            fclose(fd);
        }

        if (i==0) ctx->stats->start_time_kernel = get_time_msec(); // START CLOCK

        // 2. Send to Hailo (This blocks only if the internal queue is full)
        status = hailo_vstream_write_raw_buffer(ctx->input_vstream, buffer, ctx->frame_in_size);
        if (status != HAILO_SUCCESS) break;
    }

    free(buffer);
    
    // Important: Flush the stream so the output thread knows we are done
    hailo_flush_input_vstream(ctx->input_vstream);
    
    return NULL;
}

/** output_thread_func(void* arg)
 * 
 *  This is the output thread. Reads the result when it's available thanks to 
 *  hailo_vstream_read_raw_buffer and updates the stats.
 **/

void* output_thread_func(void* arg) {
    ThreadContext* ctx = (ThreadContext*)arg;
    hailo_status status;
    
    // Allocate buffer for results
    uint8_t* buffer = malloc(ctx->frame_out_size);

    for (int i = 0; i < ctx->total_images; i++) {
        // 1. Read Result (This waits efficiently for data)
        status = hailo_vstream_read_raw_buffer(ctx->output_vstream, buffer, ctx->frame_out_size);
        if (i==ctx->total_images-1) ctx->stats->end_time_kernel = get_time_msec(); // END CLOCK

        if (status != HAILO_SUCCESS) {
            printf("[ERROR] Read failed at frame %d\n", i);
            break;
        }

        // 2. Process Result
        int prediction = findMax(buffer);
        int expected = ctx->expected_classes[i];

        // 3. Update Stats (No mutex needed usually if only this thread writes)
        ctx->stats->total++;
        if (prediction == expected) {
            ctx->stats->correct++;
            ctx->stats->correct_count_class[prediction][prediction]++;
        } else {
            ctx->stats->correct_count_class[expected][prediction]++;
        }


    }

    ctx->stats->avg_hw_latency_per_image=ctx->stats->end_time_kernel - ctx->stats->start_time_kernel; 
    ctx->stats->avg_hw_latency_per_batch=ctx->stats->end_time_kernel - ctx->stats->start_time_kernel;


    free(buffer);
    return NULL;
}



/** load_dataset_list(const char* dir_path, ThreadContext* ctx)
 * 
 *  This function is useful to initialize the matrix of filepaths in ThreadContext structures. 
 *  Allows to set all the image paths in the dedicated matrics, them, they will be accessed by the Input Thread
 *  in order to perform the inference. 
 **/
void load_dataset_list(const char* dir_path, ThreadContext* ctx) {
    struct dirent **class_list;
    int n = scandir(dir_path, &class_list, NULL, alphasort);
    if (n < 0) return;

    int current_idx = 0;
    
    for (int i = 0; i < n; i++) {
        struct dirent *class_entry = class_list[i];
        if (class_entry->d_name[0] == '.') { free(class_entry); continue; }

        char class_path[1024];
        snprintf(class_path, sizeof(class_path), "%s/%s", dir_path, class_entry->d_name);
        
        // Scan files inside class folder
        DIR *dp = opendir(class_path);
        if (dp) {
            struct dirent *file_entry;
            while ((file_entry = readdir(dp))) {
                if (file_entry->d_name[0] == '.') continue;
                
                if (ctx->total_images >= DATASET_MAX_IMAGES) break;

                snprintf(ctx->file_paths[ctx->total_images], MAX_PATH_LEN, "%s/%s", class_path, file_entry->d_name);
                ctx->expected_classes[ctx->total_images] = current_idx; // Store the expected class index
                ctx->total_images++;
            }
            closedir(dp);
        }
        current_idx++;
        free(class_entry);
    }
    free(class_list);
}

/** readCSVFileForMemory(const char* trace_file_path, const chaer* model_path, MemoryStats* metrics)
 *  This utility function, gathers the data needed to fill the MemoryStats struct from the csv file 
 *  with path specified in trace_file_path variable.
 **/
void readCSVFileForMemory(const char* trace_file_path, const char* model_path, MemoryStats* metrics){
    FILE* fd = fopen(trace_file_path, "r");

    if (fd==NULL){
        printf("[ERROR] Error opening tracing file.\n");
        exit(1);
    }

    //model name
    char* temp_model_path=strdup(model_path);
    
    char* token=strtok(temp_model_path, "/");

    char* last_token = token;

    while(token!=NULL){
        last_token=token;
        token=strtok(NULL, "/");
    }

    char* model_name_clean = strtok(last_token, ".");

    char line[4096];
    while(fgets(line, sizeof(line), fd)){
        line[strcspn(line, "\r\n")] =0; //Removes newlines

        char* temp_line=strdup(line);
        char* first_col = strtok(temp_line, ",");

        if(!first_col){
            free(temp_line); //skip empty rows
            continue;
        }

        if (strcmp(first_col, model_name_clean) == 0){
            char* token;
            int col_idx =0;
            char* ptr = line;
            while((token = strsep(&ptr, ",")) !=NULL){
                if (col_idx==5) metrics->weights = atof(token);
                if (col_idx==33) metrics->l2_data_usage = atof(token);
                col_idx++;
            }

        }

        if(strcmp(first_col, "contexts_total") == 0){
            char* token;
            int col_idx =0;
            char* ptr = line;
            while((token = strsep(&ptr, ",")) !=NULL){
                if (col_idx==26) metrics->boundary_in = atof(token);
                if (col_idx==27) metrics->boundary_out = atof(token);
                if (col_idx==28) metrics->inter_in = atof(token);
                if (col_idx==29) metrics->inter_out = atof(token);
                col_idx++;
            }
        }

        free(temp_line);
    }

    free(temp_model_path);
    fclose(fd);
}


/** getHumanReadableValue(float bytes)
 *  Given a float value that represents bytes, returns the corresponding value in MegaBytes.
 **/
float getHumanReadableValue(float bytes){
    return ((bytes/1024)/1024);
}


/** printResults(TimeStats*, MemoryStats*)
 *  Prints the final results thanks to values contained in TimeStats and MemoryStats structs. 
 **/
void printResults(TimeStats* time_stats_values, MemoryStats* memory_stats_values){
    double accuracy = (double)time_stats_values->correct / time_stats_values->total * 100.0;
        printf("\n==================== TIME ==================\n");
        if (time_stats_values->batch_size == 1)  {
            printf("FINAL_ACCURACY: %.2f%% (%d/%d) \n", accuracy, time_stats_values->correct, time_stats_values->total); 
        } else {
            printf("FINAL_ACCURACY: %.2f%% on %d batches of %d images. \n", accuracy, time_stats_values->total/time_stats_values->batch_size, time_stats_values->batch_size);
        } 
        printf("AVG_OVERALL_LATENCY_PER_IMAGE: %f ms\n", time_stats_values->avg_overall_latency_per_image);
        printf("AVG_OVERALL_LATENCY_PER_BATCH: %f ms\n", time_stats_values->avg_overall_latency_per_batch);
        printf("AVG_HW_LATENCY_PER_IMAGE: %f ms\n", time_stats_values->avg_hw_latency_per_image);
        printf("AVG_HW_LATENCY_PER_BATCH: %f ms\n", time_stats_values->avg_hw_latency_per_batch);
        printf("TOTAL_OVERALL_LATENCY: %f ms\n", time_stats_values->overall_latency_ms);
        printf("=================== MEMORY =================\n");
        printf("MEMORY_PEAK_HOST: %.2f MB\n", getHumanReadableValue(memory_stats_values->boundary_in + memory_stats_values->boundary_out + memory_stats_values->inter_in + memory_stats_values->inter_out));
        printf("MEMORY_PEAK_ACCELERATOR: %.2f MB\n", getHumanReadableValue(memory_stats_values->weights + memory_stats_values->l2_data_usage));
        printf("==========================================\n\n");
}


/** findMaxLen(const char* classes_val[])
 *  This utility function takes the classes_names array to find the maximum length
 *  between all the class names. This is useful to printConfMatrix function.
 **/
int findMaxLen(const char * classes_val[]){
    int max_len = (int)strlen(classes_val[0]);

    for(int i=1; i<CLASSES; i++){
        if((int) strlen(classes_val[i])>max_len){
            max_len = strlen(classes_val[i]);
        }
    }

    if (max_len<12){
        max_len=12;
    }

    return max_len+5;
}

/** printConfMatrix(TimeStats*)
 *  Prints the final confusion matrix thanks to values contained in TimeStats struct. 
 **/
void printConfMatrix(TimeStats* time_stats){

    printf("\n==================== CONFUSION MATRIX ==================\n");
    int width = findMaxLen(class_names);
    printf("%-*s", width, "True \\ Pred");
    for (int i =0; i<CLASSES; i++){
        printf("%-*s", width, class_names[i]);
    }
    printf("\n");
    for(int i=0; i<CLASSES; i++){
        printf("%-*s", width, class_names[i]);
        for(int j =0; j<CLASSES; j++){
            printf("%-*d", width, time_stats->correct_count_class[i][j]);
        }
        printf("\n");
    }
    printf("========================================================\n\n");
}



/** main(int, const char**)
* 
* Initially checks if all the required parameters are correctly provided (model
* .hef file path, dataset path, batch-size). 
* After, there is an initalization section of the Hailo device with all the required variables.
* In the end, inference is performed thanks to two threads (in a streaming manner) and the final stats will be shown. 
* 
**/
int main(int argc, const char** argv) {
    if (argc<5 || sizeof(argv[1])==0 || sizeof(argv[2])==0 || sizeof(argv[3])==0 || sizeof(argv[4])==0) {
        printf("Incorrect usage.\nUsage: ./inference <path-to-model> <path-to-bin-files> <path-to-trace-mem-file> <batch-size (1-...)> \n");
        exit(1);
    }

    printf("[INFO] DATASET: %s\n", argv[2]);

    hailo_status status;
    hailo_vdevice vdevice = NULL;
    hailo_hef hef = NULL;
    hailo_configure_params_t config_params = {0};
    hailo_configured_network_group network_group = NULL;
    size_t network_group_size = 1;


    //CREATING A DEVICE TO SET THE THROTTLING OFF -----
    hailo_device_id_t * device_ids= NULL;
    size_t device_count = 1;
    hailo_device device = NULL;
    hailo_scan_devices_params_t* params= NULL;

    status = hailo_create_device_by_id(&device_ids[0], &device);


    device_ids = (hailo_device_id_t *) malloc(device_count*sizeof(hailo_device_id_t));
    status=hailo_scan_devices(params, device_ids, &device_count);
    checkStatus(status, "[INFO] DEVICE FOUND!");

    status = hailo_create_device_by_id(&device_ids[0], &device);
    checkStatus(status, "[INFO] DEVICE CREATED!");

    status = hailo_set_throttling_state(device, false);
    checkStatus(status, "[INFO] THROTTLING DISABLED!");

    bool throttling_state;
    hailo_get_throttling_state(device, &throttling_state);
    if(throttling_state==false){
        printf("[INFO] CHECK THROTTLING DISBALED PASSED!\n");
    } else {
        printf("[INFO] CHECK THROTTLING DISBALED NOT PASSED!\n");
    }

    //---------------------------------------------------

    checkStatus(hailo_create_vdevice(NULL, &vdevice), "Create VDevice");
    checkStatus(hailo_create_hef_file(&hef, argv[1]), "Load HEF");
    
    hailo_init_configure_params_by_vdevice(hef, vdevice, &config_params);
    // Set Ultra Performance Mode
    for(size_t i=0; i<config_params.network_group_params_count; i++) {
        config_params.network_group_params[i].power_mode = HAILO_POWER_MODE_ULTRA_PERFORMANCE;
    }

    checkStatus(hailo_configure_vdevice(vdevice, hef, &config_params, &network_group, &network_group_size), "Configure Device");

    hailo_input_vstream_params_by_name_t input_params[MAX_EDGE_LAYERS] = {0};
    hailo_output_vstream_params_by_name_t output_params[MAX_EDGE_LAYERS] = {0};
    size_t input_cnt = MAX_EDGE_LAYERS, output_cnt = MAX_EDGE_LAYERS;
    bool unused = false;

    hailo_make_input_vstream_params(network_group, unused, HAILO_FORMAT_TYPE_AUTO, input_params, &input_cnt);
    hailo_make_output_vstream_params(network_group, unused, HAILO_FORMAT_TYPE_AUTO, output_params, &output_cnt);

    hailo_input_vstream input_vstreams[MAX_EDGE_LAYERS];
    hailo_output_vstream output_vstreams[MAX_EDGE_LAYERS];

    checkStatus(hailo_create_input_vstreams(network_group, input_params, input_cnt, input_vstreams), "Create Input VStream");
    checkStatus(hailo_create_output_vstreams(network_group, output_params, output_cnt, output_vstreams), "Create Output VStream");

    
    ThreadContext ctx = {0};
    ctx.input_vstream = input_vstreams[0];   // Assuming single input
    ctx.output_vstream = output_vstreams[0]; // Assuming single output
    ctx.network_group = network_group;
    
    // Get frame sizes
    hailo_get_input_vstream_frame_size(ctx.input_vstream, &ctx.frame_in_size);
    hailo_get_output_vstream_frame_size(ctx.output_vstream, &ctx.frame_out_size);

    TimeStats stats = {atoi(argv[4]), 0, 0, {{0}}, 0, 0, 0, 0, 0};
    ctx.stats = &stats;
    MemoryStats memory_stats_values = {0};
    readCSVFileForMemory(argv[3], argv[1], &memory_stats_values);


    printf("[INFO] LOADING DATASET FILE LIST...\n");
    load_dataset_list(argv[2], &ctx);
    printf("[INFO] FOUND %d IMAGES. STARTING ASYNC INFERENCE...\n", ctx.total_images);

    pthread_t thread_in, thread_out;

    stats.start_time = get_time_msec(); // START CLOCK

    pthread_create(&thread_in, NULL, input_thread_func, &ctx);
    pthread_create(&thread_out, NULL, output_thread_func, &ctx);

    // 5. WAIT FOR THREADS
    pthread_join(thread_in, NULL);
    pthread_join(thread_out, NULL);

    stats.end_time = get_time_msec(); // STOP CLOCK

    // 6. PRINT RESULTS
    double total_time = stats.end_time - stats.start_time;
    stats.avg_overall_latency_per_image=total_time / stats.total; 
    stats.avg_overall_latency_per_batch=total_time / (stats.total / stats.batch_size);
    stats.avg_hw_latency_per_image = stats.avg_hw_latency_per_image / stats.total;
    stats.avg_hw_latency_per_batch = stats.avg_hw_latency_per_batch / (stats.total / stats.batch_size);

    stats.overall_latency_ms = total_time;

    
    if (stats.total > 0) {
        printResults(&stats, &memory_stats_values);
        printConfMatrix(&stats);
    }


    // Cleanup
    hailo_release_input_vstreams(input_vstreams, input_cnt);
    hailo_release_output_vstreams(output_vstreams, output_cnt);
    hailo_release_hef(hef);
    hailo_release_vdevice(vdevice);

    return 0;
}