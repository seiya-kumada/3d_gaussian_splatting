#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "util.h"
#include <boost/format.hpp>

Printer::Printer(bool quiet)
    : quiet_{quiet}
{
}
void Printer::print(const std::string &msg) const
{
    if (quiet_)
    {
        std::cout << boost::format("%1%\n") % msg;
    }
    else
    {
        // 現在の日付と時刻を取得
        boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();

        // 出力
        std::cout << boost::format("%1% [%2%]\n") % msg % now;
    }
};

void initialize_random_seed()
{
    // initialize random seed
    srand(time(NULL));
    // TODO: set up CUDA device
}