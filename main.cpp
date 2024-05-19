#include <iostream>
#include <string>

struct sConfig
{
    uint32_t initialMaxTokens;
    uint32_t expectedMaxTokens;
    uint32_t maxIterations;
    std::string tokenBlocksPath;
    uint32_t pageEndToken;
};

sConfig *parseConfig(int argc, const char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <initialMaxTokens> <expectedMaxTokens> <maxIterations> <tokenBlocksPath>" << std::endl;
        return nullptr;
    }

    sConfig *config = new sConfig;
    config->initialMaxTokens = std::stoi(argv[1]);
    config->expectedMaxTokens = std::stoi(argv[2]);
    config->maxIterations = std::stoi(argv[3]);
    config->tokenBlocksPath = argv[4];

    config->pageEndToken = config->expectedMaxTokens;

    // Print config
    std::cout << "initialMaxTokens: " << config->initialMaxTokens << std::endl;
    std::cout << "expectedMaxTokens: " << config->expectedMaxTokens << std::endl;
    std::cout << "maxIterations: " << config->maxIterations << std::endl;
    std::cout << "tokenBlocksPath: " << config->tokenBlocksPath << std::endl;

    return config;
}

int main(int argc, const char *argv[])
{
    auto config = parseConfig(argc, argv);

    return 0;
}