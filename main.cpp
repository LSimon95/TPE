#include <iostream>
#include <string>
#include <thread>
#include <filesystem>
#include <vector>
#include <map>
#include <unordered_map>
#include <cstring>
#include <fstream>

struct sConfig
{
    uint32_t initialMaxTokens;
    uint32_t expectedMaxTokens;
    uint32_t maxIterations;
    int maxWorkers;
    int tokenPairLimit;
};

struct sToken
{
    uint32_t freq;
    std::vector<uint32_t> originTokens;
};

struct sTokenBufferPage
{
    uint32_t *data;
    long size;

    sTokenBufferPage *next;
};

sConfig *parseConfig(int argc, const char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <initialMaxTokens> <expectedMaxTokens> <maxIterations> <tokenPairLimit>" << std::endl;
        exit(1);
    }

    sConfig *config = new sConfig;
    config->initialMaxTokens = std::stoi(argv[1]);
    config->expectedMaxTokens = std::stoi(argv[2]);
    config->maxIterations = std::stoi(argv[3]);
    config->tokenPairLimit = std::stoi(argv[4]);
    config->maxWorkers = std::thread::hardware_concurrency() - 1;

    // Print config
    std::cout << "initialMaxTokens: " << config->initialMaxTokens << std::endl;
    std::cout << "expectedMaxTokens: " << config->expectedMaxTokens << std::endl;
    std::cout << "maxIterations: " << config->maxIterations << std::endl;
    std::cout << "tokenPairLimit: " << config->tokenPairLimit << std::endl;

    return config;
}
void saveVocabulary(sToken **tokenTable, std::map<std::vector<uint32_t>, uint32_t> *tokenRMap, long tokenTableSize, int iteration)
{
    uint32_t totalTokensAct = 0;
    for (long i = 0; i < tokenTableSize; i++)
        if (tokenTable[i] != nullptr && tokenTable[i]->freq > 0)
            totalTokensAct++;
    std::cout << "Vocabulary size: " << totalTokensAct << std::endl;

    std::ofstream f("out/vocabulary_" + std::to_string(iteration) + ".txt");
    int tokenCnt = 0;
    for (const auto &it : *tokenRMap)
    {
        auto newtoken = it.second;
        if (tokenTable[newtoken] && tokenTable[newtoken]->freq > 0)
        {
            f << tokenCnt;
            f << " " << tokenTable[newtoken]->freq;
            for (auto &ot : it.first)
                f << " " << ot;
            f << std::endl;
            tokenCnt++;
        }
    }
}

void printProgress(long current, long total, std::string message)
{
    static long lastProgress = 0;
    long progress = (current * 100) / total;
    float ratio = (float)current / (float)total;
    if (progress != lastProgress)
    {
        std::cout << "\r" << message << " " << progress << "% (" << current << "/" << total << ")";
        std::cout.flush();
        lastProgress = progress;
    }
}

void printMaxFreqNTokens(sToken **tokenTable, long tokenTableSize, long n)
{
    std::multimap<long, uint32_t> freqMap;
    for (long i = 0; i < tokenTableSize; i++)
        if (tokenTable[i] != nullptr)
            freqMap.insert({tokenTable[i]->freq, i});

    long count = 0;
    for (auto it = freqMap.rbegin(); it != freqMap.rend(); it++)
    {
        if (count >= n)
            break;
        std::cout << "Token: " << it->second << " Freq: " << it->first << std::endl;
        count++;
    }
}

void updateTokenBuffer(int rank, sTokenBufferPage *buffer, long totalPage, std::unordered_map<uint64_t, long> *tokenPairMap, uint64_t *tokenPairReplace, uint32_t tokenPairReplaceToken, int tokenPairLimit)
{

    sTokenBufferPage *page = buffer;
    long pageCnt = 0;

    if (tokenPairReplace)
    {
        while (page)
        {
            // Replace token pair

            for (long i = 0; i < page->size - 1; i++)
            {
                uint64_t *tokenPair = (uint64_t *)(page->data + i);
                if (*tokenPair == *tokenPairReplace)
                {

                    page->data[i] = tokenPairReplaceToken;
                    if (i < (page->size - 2))
                        std::memcpy(page->data + i + 1, page->data + i + 2, (page->size - i - 2) * sizeof(uint32_t));
                    page->size--;
                }
            }
            if (rank == 0)
                printProgress(pageCnt, totalPage, "Update token buffer ");
            pageCnt++;
            page = page->next;
        }
    }

    pageCnt = 0;
    page = buffer;
    tokenPairMap->clear();
    while (page)
    {
        for (long i = 0; i < page->size - 1; i++)
        {
            uint64_t *tokenPair = (uint64_t *)(page->data + i);
            if (tokenPairMap->find(*tokenPair) == tokenPairMap->end())
                (*tokenPairMap)[*tokenPair] = 1;
            else
                tokenPairMap->at(*tokenPair)++;
        }
        if (rank == 0)
            printProgress(pageCnt, totalPage, "Scan token buffer ");
        pageCnt++;
        page = page->next;
    }

    // Get top k token pair
    std::multimap<long, uint64_t> freqMap;

    for (auto &it : *tokenPairMap)
    {
        freqMap.insert({it.second, it.first});
        if (freqMap.size() > tokenPairLimit)
            freqMap.erase(freqMap.begin());
    }
    tokenPairMap->clear();
    for (const auto &it : freqMap)
    {
        (*tokenPairMap)[it.second] = it.first;
    }
}

void getTokenBuffer(int rank, sTokenBufferPage **buffer, long *totalPage, std::unordered_map<uint64_t, long> *tokenPairMap, std::vector<std::string> orginTokensFiles)
{
    tokenPairMap->clear();

    sTokenBufferPage *lastPage = nullptr;
    long totalTokens = 0;
    for (const auto &originTokensFile : orginTokensFiles)
    {
        auto f = std::fopen(originTokensFile.c_str(), "rb");

        if (f == nullptr)
        {
            std::cerr << "Error opening file: " << originTokensFile << std::endl;
            exit(1);
        }

        uint32_t size;
        while (true)
        {
            size_t read_size = fread(&size, sizeof(uint32_t), 1, f);
            if (read_size == 0)
                break;

            auto page = new sTokenBufferPage;
            (*totalPage)++;
            if (lastPage)
                lastPage->next = page;
            else
                *buffer = page;
            page->data = new uint32_t[size];
            page->size = size;
            totalTokens += size;
            page->next = nullptr;
            lastPage = page;

            fread(page->data, sizeof(uint32_t), size, f);
        }

        fclose(f);
    }

    long readedTokens = 0;
    sTokenBufferPage *page = *buffer;
    while (page)
    {
        for (long i = 0; i < page->size; i++)
        {
            uint64_t tokenPair = page->data[i];
            if (tokenPairMap->find(tokenPair) == tokenPairMap->end())
                (*tokenPairMap)[tokenPair] = 1;
            else
                tokenPairMap->at(tokenPair)++;
        }

        readedTokens += page->size;
        if (rank == 0)
        {
            printProgress(readedTokens, totalTokens, "Worker " + std::to_string(rank) + " readed tokens");
        }
        page = page->next;
    }
}

int main(int argc, const char *argv[])
{
    auto config = parseConfig(argc, argv);

    // mkdir out
    std::filesystem::create_directory("out");

    // get all tokens files
    std::cout << "Tokens files:" << std::endl;
    std::vector<std::string> originTokensFiles;
    for (const auto &entry : std::filesystem::directory_iterator("./tokens"))
    {
        originTokensFiles.push_back(entry.path().string());
        std::cout << entry.path() << std::endl;
    }

    config->maxWorkers = std::min(config->maxWorkers, (int)originTokensFiles.size());
    config->maxWorkers = std::max(config->maxWorkers, 2);
    std::cout << "Workers: " << config->maxWorkers << std::endl;
    std::vector<std::string> originTokensFilesWorker[config->maxWorkers];

    for (int i = 0; i < originTokensFiles.size(); i++)
    {
        originTokensFilesWorker[i % config->maxWorkers].push_back(originTokensFiles[i]);
    }

    // Create token table
    long tokenTableSize = config->initialMaxTokens + config->maxIterations;
    sToken **tokenFreqTable = new sToken *[tokenTableSize];
    std::memset(tokenFreqTable, 0, sizeof(sToken *) * tokenTableSize);
    std::map<std::vector<uint32_t>, uint32_t> tokenRMap;

    for (long i = 0; i < config->initialMaxTokens; i++)
    {
        tokenFreqTable[i] = new sToken;
        tokenFreqTable[i]->freq = 0;
        tokenFreqTable[i]->originTokens.push_back(i);
        tokenRMap[{tokenFreqTable[i]->originTokens}] = i;
    }

    // Start iteration
    std::unordered_map<uint64_t, long> tokenPairFreqMap[config->maxWorkers];
    sTokenBufferPage *tokensBuffer[config->maxWorkers];
    long totalPages[config->maxWorkers];
    std::memset(totalPages, 0, sizeof(long) * config->maxWorkers);

    std::cout << "Read tokens files" << std::endl;
    std::thread *workers[config->maxWorkers];
    for (int i = 0; i < config->maxWorkers; i++)
    {
        workers[i] = new std::thread(getTokenBuffer, i, &tokensBuffer[i], (long *)(&totalPages[i]), &tokenPairFreqMap[i], originTokensFilesWorker[i]);
    }

    long totalTokens = 0;
    for (int i = 0; i < config->maxWorkers; i++)
    {
        workers[i]->join();
        auto page = tokensBuffer[i];
        while (page)
        {
            totalTokens += page->size;
            page = page->next;
        }

        delete workers[i];
    }

    for (int i = 0; i < config->maxWorkers; i++)
        for (auto &it : tokenPairFreqMap[i])
            // std::cout << it.first << " " << it.second << std::endl;
            tokenFreqTable[it.first]->freq += it.second;

    std::cout << " Total tokens: " << totalTokens << std::endl;
    printMaxFreqNTokens(tokenFreqTable, tokenTableSize, 10);
    saveVocabulary(tokenFreqTable, &tokenRMap, tokenTableSize, 0);

    uint64_t tokenPairReplace = 0;
    uint32_t tokenPairReplaceToken = 0;
    std::unordered_map<uint64_t, long> tokenPairFreqMapMerged;
    for (int i = 0; i < config->maxIterations; i++)
    {

        std::cout << "************************ Start iteration " << i << " ************************" << std::endl;
        // Find token pair to replace
        for (int j = 0; j < config->maxWorkers; j++)
        {
            workers[j] = new std::thread(
                updateTokenBuffer,
                j,
                tokensBuffer[j],
                totalPages[j],
                tokenPairFreqMap + j,
                i == 0 ? nullptr : &tokenPairReplace,
                tokenPairReplaceToken,
                config->tokenPairLimit);
        }

        for (int j = 0; j < config->maxWorkers; j++)
        {
            workers[j]->join();
            delete workers[j];
        }

        // Merge token pair freq map
        tokenPairFreqMapMerged.clear();
        for (int j = 0; j < config->maxWorkers; j++)
            for (auto &it : tokenPairFreqMap[j])
                if (tokenPairFreqMapMerged.find(it.first) == tokenPairFreqMapMerged.end())
                    tokenPairFreqMapMerged[it.first] = it.second;
                else
                    tokenPairFreqMapMerged[it.first] += it.second;

        uint64_t maxFreqToken = 0;
        uint32_t maxFreq = 0;

        if (tokenPairFreqMapMerged.size() == 0)
        {
            std::cout << "No token pair to replace" << std::endl;
            break;
        }

        for (auto &it : tokenPairFreqMapMerged)
            if (it.second > maxFreq)
            {
                maxFreq = it.second;
                maxFreqToken = it.first;
            }

        uint32_t curToken = config->initialMaxTokens + i;
        tokenFreqTable[curToken] = new sToken;
        tokenFreqTable[curToken]->freq = maxFreq;

        auto curOriginTokens = &(tokenFreqTable[curToken]->originTokens);

        uint32_t *pt12 = (uint32_t *)&maxFreqToken;
        *curOriginTokens = tokenFreqTable[pt12[0]]->originTokens;
        tokenFreqTable[pt12[0]]->freq -= maxFreq;
        auto originTokens2 = &(tokenFreqTable[pt12[1]]->originTokens);
        curOriginTokens->insert(curOriginTokens->end(), originTokens2->begin(), originTokens2->end());
        tokenFreqTable[pt12[1]]->freq -= maxFreq;

        std::cout << std::endl;
        std::cout << "Max freq token pair (" << maxFreq << "):";
        for (auto &it : *curOriginTokens)
            std::cout << it << " ";
        std::cout << std::endl;

        tokenRMap[*curOriginTokens] = curToken;

        tokenPairReplace = maxFreqToken;
        tokenPairReplaceToken = curToken;

        // Some satistics
        uint32_t totalTokensAct = 0;
        for (int i = 0; i < tokenTableSize; i++)
            if (tokenFreqTable[i] != nullptr && tokenFreqTable[i]->freq > 0)
                totalTokensAct++;

        saveVocabulary(tokenFreqTable, &tokenRMap, tokenTableSize, i + 1);
        std::cout << "************************ End iteration " << i << " ************************" << std::endl;
    }

    // Free memory
    for (int i = 0; i < config->maxWorkers; i++)
    {
        auto page = tokensBuffer[i];
        while (page)
        {
            auto next = page->next;
            delete[] page->data;
            delete page;
            page = next;
        }
    }

    delete config;
    for (long i = 0; i < tokenTableSize; i++)
        if (tokenFreqTable[i] != nullptr)
            delete tokenFreqTable[i];
    delete[] tokenFreqTable;

    return 0;
}