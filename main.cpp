#include <iostream>
#include <string>
#include <thread>
#include <filesystem>
#include <vector>
#include <map>
#include <cstring>

struct sConfig
{
    uint32_t initialMaxTokens;
    uint32_t expectedMaxTokens;
    uint32_t maxIterations;
    int maxWorkers;
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
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <initialMaxTokens> <expectedMaxTokens> <maxIterations>" << std::endl;
        exit(1);
    }

    sConfig *config = new sConfig;
    config->initialMaxTokens = std::stoi(argv[1]);
    config->expectedMaxTokens = std::stoi(argv[2]);
    config->maxIterations = std::stoi(argv[3]);
    config->maxWorkers = std::thread::hardware_concurrency() - 1;

    // Print config
    std::cout << "initialMaxTokens: " << config->initialMaxTokens << std::endl;
    std::cout << "expectedMaxTokens: " << config->expectedMaxTokens << std::endl;
    std::cout << "maxIterations: " << config->maxIterations << std::endl;

    return config;
}

void printProgress(long current, long total, std::string message)
{
    static long lastProgress = 0;
    long progress = (current * 100) / total;
    float ratio = (float)current / (float)total;
    if (progress != lastProgress)
    {
        std::cout << "\r" << message << " " << progress << "% (" << current << "/" << total << ")" << std::endl;
        lastProgress = progress;
    }
}

void printMaxFreqNTokens(sToken **tokenTable, long tokenTableSize, long n)
{
    std::multimap<uint64_t, uint32_t> freqMap;
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

void updateTokenBuffer(std::vector<sTokenBufferPage> *buffer, std::map<uint64_t, long> *tokenPairMap, uint64_t *tokenPairReplace)
{
    tokenPairMap->clear();
}

void getTokenBuffer(int rank, sTokenBufferPage **buffer, std::map<uint64_t, long> *tokenPairMap, std::vector<std::string> orginTokensFiles)
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

    // get all tokens files
    std::cout << "Tokens files:" << std::endl;
    std::vector<std::string> originTokensFiles;
    for (const auto &entry : std::filesystem::directory_iterator("./tokens"))
    {
        originTokensFiles.push_back(entry.path().string());
        std::cout << entry.path() << std::endl;
    }

    config->maxWorkers = std::min(config->maxWorkers, (int)originTokensFiles.size());
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
    std::map<uint64_t, long> tokenPairFreqMap[config->maxWorkers];
    sTokenBufferPage *tokensBuffer[config->maxWorkers];

    std::cout << "Read tokens files" << std::endl;
    std::thread *workers[config->maxWorkers];
    for (int i = 0; i < config->maxWorkers; i++) {
        workers[i] = new std::thread(getTokenBuffer, i, &tokensBuffer[i], &tokenPairFreqMap[i], originTokensFilesWorker[i]);
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

    std::cout << "Total tokens: " << totalTokens << std::endl;
    printMaxFreqNTokens(tokenFreqTable, tokenTableSize, 10);

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

    // End iteration

    // Free memory
    delete config;
    for (long i = 0; i < tokenTableSize; i++)
        if (tokenFreqTable[i] != nullptr)
            delete tokenFreqTable[i];
    delete[] tokenFreqTable;

    return 0;
}