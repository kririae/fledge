#include "tui.hpp"

#include "fmt/core.h"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/screen.hpp"
#include "ftxui/screen/string.hpp"

FLG_NAMESPACE_BEGIN
namespace tui {

// Only a sample for now
void MakeWindow() {
  using namespace ftxui;

  auto summary = [&] {
    auto content = vbox({
        hbox({text(L"- done:   "), text(L"3") | bold}) | color(Color::Green),
        hbox({text(L"- active: "), text(L"2") | bold}) | color(Color::RedLight),
        hbox({text(L"- queue:  "), text(L"9") | bold}) | color(Color::Red),
    });
    return window(text(L" Summary "), content);
  };

  auto document =  //
      vbox({
          hbox({
              summary(),
              summary(),
              summary() | flex,
          }),
          summary(),
          summary(),
      });

  // Limit the size of the document to 80 char.
  document = document | size(WIDTH, LESS_THAN, 80);

  auto screen = Screen::Create(Dimension::Full(), Dimension::Fit(document));

  ftxui::Render(screen, document);
  fmt::print("{}\0", screen.ToString());
}

}  // namespace tui
FLG_NAMESPACE_END
