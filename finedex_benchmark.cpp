#include "util.h"
#include "aidel.h"
#include "aidel_impl.h"

#define CACHELINE_SIZE (1 << 6)

struct alignas(CACHELINE_SIZE) ThreadParam;

typedef int64_t key_type;
typedef int64_t val_type;
typedef aidel::AIDEL<key_type, val_type> aidel_type;
typedef ThreadParam thread_param_t;


struct alignas(CACHELINE_SIZE) ThreadParam {
    aidel_type *ai;
    uint64_t throughput;
    uint32_t thread_id;
};


// parameters
double read_ratio = 0;
double insert_ratio = 1;
double update_ratio = 0;
double delete_ratio = 0;
size_t item_num  = 20000000;
size_t exist_num = 20000000;
size_t runtime = 5;
size_t thread_num = 1;
size_t benchmark = 0;  

volatile bool running = false;
std::atomic<size_t> ready_threads(0);
std::vector<key_type> exist_keys;
std::vector<key_type> non_exist_keys;


inline void prepare(aidel_type *&ai);
void run_benchmark(aidel_type *ai, size_t sec);
void *run_fg(void *param);
inline void parse_args(int, char **);
void check(aidel_type *ai);

void normal_data();
void lognormal_data();



int main(int argc, char **argv) {
    parse_args(argc, argv);
    aidel_type *ai;
    prepare(ai);
    run_benchmark(ai, runtime);
    ai->self_check();
    if(ai!=nullptr) delete ai;
}

inline void parse_args(int argc, char **argv) {
  struct option long_options[] = {
      {"read", required_argument, 0, 'a'},
      {"insert", required_argument, 0, 'b'},
      {"remove", required_argument, 0, 'c'},
      {"update", required_argument, 0, 'd'},
      {"item_num", required_argument, 0, 'e'},
      {"runtime", required_argument, 0, 'f'},
      {"thread_num", required_argument, 0, 'g'},
      {"benchmark", required_argument, 0, 'h'},
      {0, 0, 0, 0}};
  std::string ops = "a:b:c:d:e:f:g:h:";
  int option_index = 0;

  while (1) {
    int c = getopt_long(argc, argv, ops.c_str(), long_options, &option_index);
    if (c == -1) break;

    switch (c) {
      case 0:
        if (long_options[option_index].flag != 0) break;
        abort();
        break;
      case 'a':
        read_ratio = strtod(optarg, NULL);
        INVARIANT(read_ratio >= 0 && read_ratio <= 1);
        break;
      case 'b':
        insert_ratio = strtod(optarg, NULL);
        INVARIANT(insert_ratio >= 0 && insert_ratio <= 1);
        break;
      case 'c':
        delete_ratio = strtod(optarg, NULL);
        INVARIANT(delete_ratio >= 0 && delete_ratio <= 1);
        break;
      case 'd':
        update_ratio = strtod(optarg, NULL);
        INVARIANT(update_ratio >= 0 && update_ratio <= 1);
        break;
      case 'e':
        item_num = strtoul(optarg, NULL, 10);
        INVARIANT(item_num > 0);
        break;
      case 'f':
        runtime = strtoul(optarg, NULL, 10);
        INVARIANT(runtime > 0);
        break;
      case 'g':
        thread_num = strtoul(optarg, NULL, 10);
        INVARIANT(thread_num > 0);
        break;
      case 'h':
        benchmark = strtoul(optarg, NULL, 10);
        INVARIANT(benchmark >= 0 && benchmark<4);
        break;
      default:
        abort();
    }
  }

  COUT_THIS("[micro] Read:Insert:Update:Delete:Scan = "
            << read_ratio << ":" << insert_ratio << ":" << update_ratio << ":"
            << delete_ratio );
  double ratio_sum =
      read_ratio + insert_ratio + delete_ratio + update_ratio;
  INVARIANT(ratio_sum > 0.9999 && ratio_sum < 1.0001);  // avoid precision lost
  COUT_VAR(runtime);
  COUT_VAR(thread_num);
  COUT_VAR(benchmark);
}

void prepare(aidel_type *&ai){
    switch (benchmark) {
        case 0:
            normal_data();
			break;
		case 1:
			lognormal_data();
			break;
		default:
			abort();
    }

    // initilize XIndex (sort keys first)
    COUT_THIS("[processing data]");
    std::sort(exist_keys.begin(), exist_keys.end());
    exist_keys.erase(std::unique(exist_keys.begin(), exist_keys.end()), exist_keys.end());
    std::sort(exist_keys.begin(), exist_keys.end());
    for(size_t i=1; i<exist_keys.size(); i++){
        assert(exist_keys[i]>=exist_keys[i-1]);
    }
    //std::vector<val_type> vals(exist_keys.size(), 1);

    COUT_VAR(exist_keys.size());
    COUT_VAR(non_exist_keys.size());
    
    COUT_THIS("[Training aidel]");
    double time_s = 0.0;
    TIMER_DECLARE(0);
    TIMER_BEGIN(0);
    size_t maxErr = 4;
    ai = new aidel_type();
    ai->train(exist_keys, exist_keys, 64);
    TIMER_END_S(0,time_s);
    printf("%8.1lf s : %.40s\n", time_s, "training");
    ai->self_check();
    COUT_THIS("check aidel: OK");
}

void run_benchmark(aidel_type *ai, size_t sec) {
    pthread_t threads[thread_num];
    thread_param_t thread_params[thread_num];
    // check if parameters are cacheline aligned
    for (size_t i = 0; i < thread_num; i++) {
        if ((uint64_t)(&(thread_params[i])) % CACHELINE_SIZE != 0) {
            COUT_N_EXIT("wrong parameter address: " << &(thread_params[i]));
        }
    }

    running = false;
    for(size_t worker_i = 0; worker_i < thread_num; worker_i++){
        thread_params[worker_i].ai = ai;
        thread_params[worker_i].thread_id = worker_i;
        thread_params[worker_i].throughput = 0;
        int ret = pthread_create(&threads[worker_i], nullptr, run_fg,
                                (void *)&thread_params[worker_i]);
        if (ret) {
            COUT_N_EXIT("Error:" << ret);
        }
    }

    COUT_THIS("[micro] prepare data ...");
    while (ready_threads < thread_num) sleep(0.5);

    running = true;
    std::vector<size_t> tput_history(thread_num, 0);
    size_t current_sec = 0;
    while (current_sec < sec) {
        sleep(1);
        uint64_t tput = 0;
        for (size_t i = 0; i < thread_num; i++) {
            tput += thread_params[i].throughput - tput_history[i];
            tput_history[i] = thread_params[i].throughput;
        }
        COUT_THIS("[micro] >>> sec " << current_sec << " throughput: " << tput);
        ++current_sec;
    }

    running = false;
    void *status;
    for (size_t i = 0; i < thread_num; i++) {
        int rc = pthread_join(threads[i], &status);
        if (rc) {
            COUT_N_EXIT("Error:unable to join," << rc);
        }
    }

    size_t throughput = 0;
    for (auto &p : thread_params) {
        throughput += p.throughput;
    }
    COUT_THIS("[micro] Throughput(op/s): " << throughput / sec);
}

void *run_fg(void *param) {
    thread_param_t &thread_param = *(thread_param_t *)param;
    uint32_t thread_id = thread_param.thread_id;
    aidel_type *ai = thread_param.ai;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> ratio_dis(0, 1);

    size_t non_exist_key_n_per_thread = non_exist_keys.size() / thread_num;
    size_t non_exist_key_start = thread_id * non_exist_key_n_per_thread;
    size_t non_exist_key_end = (thread_id + 1) * non_exist_key_n_per_thread;
    std::vector<key_type> op_keys(non_exist_keys.begin() + non_exist_key_start,
                                   non_exist_keys.begin() + non_exist_key_end);

    COUT_THIS("[micro] Worker" << thread_id << " Ready.");
    size_t query_i = 0, insert_i = 0, delete_i = 0, update_i = 0;
    // exsiting keys fall within range [delete_i, insert_i)
    ready_threads++;
    volatile result_t res = result_t::failed;
    val_type dummy_value = 1234;

    while (!running)
        ;
	while (running) {
        double d = ratio_dis(gen);
        if (d <= read_ratio) {                   // search
            key_type dummy_key = exist_keys[query_i % exist_keys.size()];
            res = ai->find(dummy_key, dummy_value);
            query_i++;
            if (unlikely(query_i == exist_keys.size())) {
                query_i = 0;
            }
        } else if (d <= read_ratio+insert_ratio){  // insert
            key_type dummy_key = non_exist_keys[insert_i % non_exist_keys.size()];
            res = ai->insert(dummy_key, dummy_key);
            insert_i++;
            if (unlikely(insert_i == non_exist_keys.size())) {
                insert_i = 0;
            }
        } else if (d <= read_ratio+insert_ratio+update_ratio) {    // update
            key_type dummy_key = non_exist_keys[update_i % non_exist_keys.size()];
            res = ai->update(dummy_key, dummy_key);
            update_i++;
            if (unlikely(update_i == non_exist_keys.size())) {
                update_i = 0;
            }
        }  else {                // remove
            key_type dummy_key = exist_keys[delete_i % exist_keys.size()];
            res = ai->remove(dummy_key);
            delete_i++;
            if (unlikely(delete_i == exist_keys.size())) {
                delete_i = 0;
            }
        }
        thread_param.throughput++;
    }
    pthread_exit(nullptr);
}

void check(aidel_type *ai){
	std::cout<<"check the correctness after inserts: ";
	result_t res = result_t::failed;
	key_type dummy_key = 1234;
	val_type dummy_val = 4321;
	for(int i = 0; i<exist_keys.size(); i++){
		dummy_key = exist_keys[i];
		res = ai->find(dummy_key, dummy_val);
		assert(res == result_t::ok);
		assert(dummy_key==dummy_val);
	}
	for(int i=0; i<non_exist_keys.size(); i++){
		dummy_key = non_exist_keys[i];
		res = ai->find(dummy_key, dummy_val);
		assert(res == result_t::ok);
		assert(dummy_key==dummy_val);
	}
	std::cout<<" OK" << std::endl;
}



void normal_data(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> rand_normal(4, 2);

    exist_keys.reserve(exist_num);
    for (size_t i = 0; i < exist_num; ++i) {
        int64_t a = rand_normal(gen)*1000000000000;
        if(a<0) {
            i--;
            continue;
        }
        exist_keys.push_back(a);
    }
    if (insert_ratio > 0) {
        non_exist_keys.reserve(item_num);
        for (size_t i = 0; i < item_num; ++i) {
            int64_t a = rand_normal(gen)*1000000000000;
            if(a<0) {
                i--;
                continue;
            }
            non_exist_keys.push_back(a);
        }
    }
}
void lognormal_data(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::lognormal_distribution<double> rand_lognormal(0, 2);

    exist_keys.reserve(exist_num);
    for (size_t i = 0; i < exist_num; ++i) {
        int64_t a = rand_lognormal(gen)*1000000000000;
        assert(a>0);
        exist_keys.push_back(a);
    }
    if (insert_ratio > 0) {
        non_exist_keys.reserve(item_num);
        for (size_t i = 0; i < item_num; ++i) {
            int64_t a = rand_lognormal(gen)*1000000000000;
            assert(a>0);
            non_exist_keys.push_back(a);
        }
    }
}


