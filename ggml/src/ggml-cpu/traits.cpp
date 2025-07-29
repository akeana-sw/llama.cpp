#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

#if 0
#define DBG_INFO(x) \
    if (params->ith == 0) std::cout << __FILE__ << " " << std::setfill(' ') << std::setw(3) << __LINE__ << " -" << x << "\n";
#else
#define DBG_INFO(x) \
    if (params->ith == 0) std::cout << x << "\n";
#endif

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

static bool ggml_compute_forward_external(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    auto debug_print_tensor = [&](struct ggml_tensor *dst, std::string name = "UNKNOWN") {
	auto append_dim = [&](int val) {
	    std::ostringstream buf;
	    buf << "";
	    if (val > 1)
		buf << "[" << val << "]";
	    return buf.str();
	};
	auto append_info = [&](struct ggml_tensor *dst) {
	    std::ostringstream buf;

	    buf << " " << ((dst->nb[0] == 4) ? "(float)" : "(_Float16)")
		<< dst->name
		<< append_dim(dst->ne[3])
		<< append_dim(dst->ne[2])
		<< append_dim(dst->ne[1])
		<< append_dim(dst->ne[0]);
	    return buf.str();
	};

	std::ostringstream buf;

	buf << append_info(dst) << " = " << name << "(";

	for (auto i = 0; dst->src[i]; i++) {
	    if (i > 0)
		buf << ",";
	    buf << append_info(dst->src[i]);
	}
	buf << " )";
	return buf.str();
    };

    auto invoke_makefile = [&](struct ggml_tensor *dst,
			       std::string func,
			       std::string args) {

	if (params->ith != 0)
	    return true;

	std::cout << debug_print_tensor(dst, func) << "\t" << std::flush;

	// Write dst->src[*] data to a file.
	auto data_to_file = [&](std::string filename, struct ggml_tensor *dst) {
	    std::ofstream fstream(filename.c_str(), std::ios::binary);
	    fstream.write(reinterpret_cast<char*>(dst->data), dst->nb[3]);
	    fstream.close();
	};

	// Copy data from output file
	auto file_to_data = [&](std::string filename, struct ggml_tensor *dst) {
	    std::ifstream fstream((filename + ".ram").c_str(), std::ios::binary);
	    fstream.seekg(128 * 512);
	    fstream.read(reinterpret_cast<char*>(dst->data), dst->nb[3]);
	    fstream.close();
	};

	// Invoke makefile
	auto invoke_make = [&](std::string path, std::string args) {
	    std::ostringstream cmd;
	    cmd << "make -s -C " << path << " VLEN=256 " << args;
	    pclose(popen(cmd.str().c_str(), "w"));
	};

	std::filesystem::path path = std::filesystem::current_path().parent_path() / "akeana";

	assert(!std::filesystem::create_directory(path));
	assert(!std::filesystem::create_directory(path / func));
	assert(!std::filesystem::create_directory(path / func / "input"));
	assert(!std::filesystem::create_directory(path / func / "output"));

#if 0
	if (func == "dotprod") {
	    for (auto i = 1; dst->src[i] != NULL; i++)
		data_to_file(path / func / "input" / dst->src[i]->name, dst->src[i]);
	} else {
	    for (auto i = 0; dst->src[i] != NULL; i++)
		data_to_file(path / func / "input" / dst->src[i]->name, dst->src[i]);
	}
#else
	for (auto i = 0; dst->src[i] != NULL; i++)
	    data_to_file(path / func / "input" / dst->src[i]->name, dst->src[i]);
#endif

	invoke_make(path, func + ".ram " + args);

	file_to_data(path / func / "output" / dst->name, dst);

	return true;
    };

    std::ostringstream cmdargs;

    switch (dst->op) {
    case GGML_OP_ADD: // 2
	// ggml_compute_forward_add
	// ggml_compute_forward_add_non_quantized

#if 0
	cmdargs << " SRC0=" << dst->src[0]->name << " DIM1=" << dst->ne[0]
		<< " SRC1=" << dst->src[1]->name << " DIM2=" << dst->ne[1]
		<< " DST=" << dst->name;

	return invoke_makefile(dst, "matadd", cmdargs.str());
#else
	// DBG_INFO(debug_print_tensor(dst, "matadd"));
#endif
	break;
    case GGML_OP_MUL:
#if 0
	if (dst->src[1]->nb[3] != dst->src[1]->nb[1]) {
	    DBG_INFO(debug_print_tensor(dst, "matmul"));
	    break;
	}
	cmdargs << " SRC0=" << dst->src[0]->name << " DIM1=" << dst->ne[0]
		<< " SRC1=" << dst->src[1]->name << " DIM2=" << dst->ne[1]
		<< " DST=" << dst->name;

	return invoke_makefile(dst, "matmul", cmdargs.str());
#else
	// DBG_INFO(debug_print_tensor(dst, "matmul"));
#endif
	break;
    case GGML_OP_RMS_NORM:	break; // DBG_INFO(debug_print_tensor(dst, "rms_norm")); break;
    case GGML_OP_MUL_MAT: // 27
	// ggml_compute_forward_mul_mat
	// ggml_compute_forward_mul_mat_one_chunk
#if 0
	// Dictionary is 128256; leave that for the host (default)
	if (dst->ne[0] > (1 << 16)) {
	    DBG_INFO(debug_print_tensor(dst, "HOST -> dotprod"));
	    break;
	}

	// TODO: Special case where the output is not one dimensional.
	// Worry about that later.
	if (dst->ne[1] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
	    DBG_INFO(debug_print_tensor(dst, "dotprod"));
	    break;
	}

	cmdargs << " SRC0=" << dst->src[0]->name << " DIM1=" << dst->src[0]->ne[0]
		<< " SRC1=" << dst->src[1]->name << " DIM2=" << dst->ne[0]
		<< " DST=" << dst->name;

	return invoke_makefile(dst, "dotprod", cmdargs.str());
#else
	// DBG_INFO(debug_print_tensor(dst, "dotprod"));
#endif
	break;
    case GGML_OP_CPY:		break; // DBG_INFO(debug_print_tensor(dst, "cp")); break;
    case GGML_OP_CONT:		break; // DBG_INFO(debug_print_tensor(dst, "cont")); break;
    case GGML_OP_RESHAPE:	break; // DBG_INFO(debug_print_tensor(dst, "reshape")); break;
    case GGML_OP_VIEW:		break; // DBG_INFO(debug_print_tensor(dst, "view")); break;
    case GGML_OP_PERMUTE:	break; // DBG_INFO(debug_print_tensor(dst, "permute")); break;
    case GGML_OP_TRANSPOSE:	break; // DBG_INFO(debug_print_tensor(dst, "transpose")); break;
    case GGML_OP_GET_ROWS:
#if 0
	if (params->ith == 0 && dst->src[0]->ne[1] > (1 << 16))
	    std::cout << "\n";
	// DBG_INFO(debug_print_tensor(dst, "get_rows"));
#endif
	break;
    case GGML_OP_SOFT_MAX:	break; // DBG_INFO(debug_print_tensor(dst, "softmax")); break;
    case GGML_OP_ROPE:		break; // DBG_INFO(debug_print_tensor(dst, "rope")); break;
    case GGML_OP_UNARY:
	switch (ggml_get_unary_op(dst)) {
	case GGML_UNARY_OP_SILU: break; // DBG_INFO(debug_print_tensor(dst, "silu")); break;
	default:		 DBG_INFO(debug_print_tensor(dst)); break;
	}
	break;
    case GGML_OP_GLU:
	switch (dst->op_params[0]) {
	case GGML_GLU_OP_SWIGLU: break; // DBG_INFO(debug_print_tensor(dst, "swiglu")); break;
	default:		 DBG_INFO(debug_print_tensor(dst)); break;
	}
	break;
    default:			 DBG_INFO(debug_print_tensor(dst)); break;
    }
    return false;
}

static void ggml_verify_forward_external(struct ggml_compute_params * params, struct ggml_tensor * op) {

    auto file_to_data = [&](std::string filename, struct ggml_tensor *dst) {
	int val[128];
	std::ifstream fstream(filename.c_str(), std::ios::binary);
	fstream.seekg(128 * 512);
	// fstream.read(reinterpret_cast<char*>(dst->data), dst->nb[3]);
	fstream.read(reinterpret_cast<char*>(val), std::min(256, (int)dst->nb[3]));
	// for (auto i = 0; i < dst->nb[3]; i++)
	for (auto i = 0; i < std::min(64, (int)dst->nb[3]); i++)
	    if (reinterpret_cast<int*>(dst->data)[i] != val[i])
		std::cout << __FILE__ << " " << std::setfill(' ') << std::setw(3) << __LINE__
			  << " " << "val[" << i << "](0x" << std::hex << val[i] << ") != 0x"
			  << reinterpret_cast<int*>(dst->data)[i] << std::dec << "\n";
	fstream.close();
    };


    switch (op->op) {
    case GGML_OP_MUL_MAT: // 27
	if (op->ne[0] < (1 << 16) &&
	    op->ne[1] == 1 && op->ne[2] == 1 && op->ne[3] == 1) {
	    if (params->ith != 0)
		break;

	    std::cout << __FILE__ << " " << std::setfill(' ') << std::setw(3) << __LINE__
		      << " " << op->nb[3] << " " << op->ne[0] << "\n";

	    std::string path = std::filesystem::current_path().parent_path().string() + "/akeana";

	    file_to_data(path + "/dotprod/output/" + op->name + ".ram", op);
	}
	break;
    default:
	break;
    }
}

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    // TODO: Can fold this into `tensor_traits->compute_forward()`
    if (ggml_compute_forward_external(params, op))
	return true;

    for (auto extra : ggml_backend_cpu_get_extra_buffers_type()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
#if 0
		ggml_verify_forward_external(params, op);
#endif
                return true;
            }
        }
    }
    return false;
}

bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
    for (auto extra : ggml_backend_cpu_get_extra_buffers_type()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                return true;
            }
        }
    }
    return false;
}
