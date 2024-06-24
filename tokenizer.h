#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <cstring>

struct ATS {
    int* tokens;
    int size;
};

struct ATNode {
    int token;
    std::multimap<int, ATNode>* next;
};

struct TokenEle {
    int token;
    int index;
};

class TPE{
    public:
        TPE(const char* vocab_file) {
            m_ats2stRoot.token = -1;
            m_ats2stRoot.next = nullptr;
            std::ifstream file(vocab_file);
            std::string line;
            while (std::getline(file, line)) {
                std::string token;
                std::vector<int> ats;
                bool first = true;
                std::multimap<int, ATS*>::iterator it;

                for (auto &c : line) {
                    if (c == ' ') {
                        if (first) {
                            first = false;
                            // Semantic token to audio token
                            m_st2ats.insert(std::pair<int, ATS*>(std::stoi(token), new ATS));
                            it = m_st2ats.find(std::stoi(token));
                        }
                        else ats.push_back(std::stoi(token));
                        token = "";
                    }
                    token += c;
                }
                ats.push_back(std::stoi(token));
                int* token_pairs_array = new int[ats.size()];
                for (int i = 0; i < ats.size(); i++) {
                    token_pairs_array[i] = ats[i];
                }
                (*it->second) = {token_pairs_array, (int)(ats.size())};

                // Audio token to semantic token
                ATNode* node = &m_ats2stRoot;
                for (int i = 0; i < ats.size(); i++) {

                    if (node->next == nullptr) node->next = new std::multimap<int, ATNode>;

                    auto itAT = node->next->find(ats[i]);
                    if (itAT == node->next->end()) {
                        node->next->insert(std::pair<int, ATNode>(ats[i], {-1, nullptr}));
                        itAT = node->next->find(ats[i]);
                    }
                    node = &itAT->second;
                }
                node->token = it->first;
            }
        };

        void deleteATNode(ATNode* node) {
            if (node->next != nullptr) {
                for (auto &it : *node->next) {
                    deleteATNode(&it.second);
                }
                delete node->next;
            }
        }
        ~TPE() {
            for (auto &it : m_st2ats) {
                delete it.second->tokens;
                delete it.second;
            }
            deleteATNode(&m_ats2stRoot);
        };

        std::multimap<int, ATS*> m_st2ats;
        ATNode m_ats2stRoot;

};
