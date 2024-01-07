#include <string>

void initialize_random_seed();
class Printer
{
private:
    bool quiet_;

public:
    Printer(bool quiet);
    void print(const std::string &msg) const;
};
