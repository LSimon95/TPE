#include "tokenizer.h"

extern "C" {
    TPE* tpe_new(const char* vocab_file){ return new TPE(vocab_file); }
    void tpe_delete(TPE* tpe){ delete tpe; }

    int* tpe_st2at(TPE* tpe, int* st, int size, int max_tokens) {
        int* at = new int[max_tokens];
        int* at_size = at;
        *at_size = 0;

        for (int i = 0; i < size; i++) {
            auto it = tpe->m_st2ats.find(st[i]);
            if (it == tpe->m_st2ats.end()) continue;
            if (*at_size + it->second->size > max_tokens) break;
            memcpy(at + *at_size + 1, it->second->tokens, it->second->size * sizeof(int));
            *at_size += it->second->size;
        }
        return at;
    }

    int* n_at_per_st(TPE* tpe, int* st, int size) {
        int* numbers = new int[size];

        for (int i = 0; i < size; i++) {
            auto it = tpe->m_st2ats.find(st[i]);
            if (it == tpe->m_st2ats.end()) {
                numbers[i] = 0;
                continue;
            }
            numbers[i] = it->second->size;
        }
        return numbers;
    }

    int* tpe_at2st(TPE* tpe, int*at, int size, int max_tokens) {
        int* st = new int[max_tokens];
        int* st_size = st;
        *st_size = 0;

        ATNode* node = nullptr;
        std::vector<TokenEle> tokenStack;
        for (int i = 0; i < size; i++) {
            if (node == nullptr) {
                auto tpeNode = tpe->m_ats2stRoot.next;
                auto it = tpeNode->find(at[i]);
                if (it == tpeNode->end()) continue;
                node = &it->second;
            }

            if (node->token != -1) tokenStack.push_back({node->token, i});

            if ((node->next != nullptr && i + 1 < size) && (node->next->find(at[i + 1]) != node->next->end())) {
                node = &node->next->find(at[i + 1])->second;
                
                if (node) continue;
            }

            int token = node->token;
            if (token == -1 && tokenStack.size() > 0) {
                token = tokenStack.back().token;
                i = tokenStack.back().index;
            }

            if (token != -1) {
                if (*st_size + 1 > max_tokens) break;
                st[++(*st_size)] = token;
            }
            node = nullptr;
            tokenStack.clear();
        }
        return st;
    }

    int get_vocab_size(TPE* tpe) { return tpe->m_st2ats.size(); }
    void free_ptr(int* ptr) { delete ptr; }
}