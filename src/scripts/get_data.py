import bs4
from selenium import webdriver
from selenium.webdriver.chrome.options import ChromiumOptions
import pandas as pd
import numpy as np

URL = "https://www.pollbludger.net/fed2025/bludgertrack/polldata.htm?"
output_filename = "src/data/Polls/poll_data_latest.csv"

options = ChromiumOptions()
options.add_argument("--headless=new")
driver = webdriver.Chrome(options=options)

print("Loading the webpage...")
driver.get(URL)
page_source = driver.page_source
driver.quit()

# Parse the HTML
print("Parsing the HTML")
soup = bs4.BeautifulSoup(page_source, "html.parser")


# Columns in the table
# everything with the r/a are the respondents preference choices
# We'll only use these when 2pp from historical preference data isn't available
columns = [
    "StartDate",
    "EndDate",
    "PollName",
    "Mode",
    "Scope",
    "Sample",
    "ALP",
    "LNP",
    "GRN",
    "PHON",
    "UND",
    "ALP_2pp",
    "LNP_2pp",
    "ALP r/a",
    "L-NP r/a",
    "UND r/a",
]

data = []
for tr in soup.find("table").find_all("tr"):
    row = [td.text for td in tr.find_all("td")]
    # If want to filter out all except LA then can do that here
    data.append(row)

# First entry seems to be the entire table? And second is blank
df = pd.DataFrame(data[2:], columns=columns)


print("Changing the column types...")
# Fix up our column types
df.loc[:, ["StartDate", "EndDate"]] = df.loc[:, ["StartDate", "EndDate"]].apply(
    pd.to_datetime, format="mixed", errors="coerce"
)
numeric_columns = [
    "ALP",
    "LNP",
    "GRN",
    "PHON",
    "UND",
    "ALP_2pp",
    "LNP_2pp",
    "ALP r/a",
    "L-NP r/a",
]
df.loc[:, numeric_columns] = df.loc[:, numeric_columns].apply(
    pd.to_numeric, errors="coerce"
)
# Now take the 2pp vote using historical preferences where available, and
# the indicated preferences otherwise
mask = np.isfinite(df["ALP_2pp"].values.astype(float))
df["ALP_2pp"] = np.where(
    mask, df["ALP_2pp"].values.astype(float), df["ALP r/a"].values.astype(float)
)
mask = np.isfinite(df["LNP_2pp"].values.astype(float))
df["LNP_2pp"] = np.where(
    mask, df["LNP_2pp"].values.astype(float), df["L-NP r/a"].values.astype(float)
)

# Have to treat the 'Sample' column differently, since it has commas in the numbers
df["Sample"] = df["Sample"].str.replace(",", "").apply(pd.to_numeric, errors="coerce")

# And finally, replace the spaces and slashes in pollster names with an underscore
df["PollName"] = df["PollName"].str.replace(" |/", "_", regex=True)

# Drop columns we don't want
df = df.drop(
    ["UND r/a", "L-NP r/a", "ALP r/a"],
    axis=1,
)

# Save the data
print(f"Saving to {output_filename}")
df.to_csv(output_filename)
print("Done!")