import random
import pandas as pd

def generate_dataset(n=100):
    # Lists of possible values
    education_levels = ["High School", "Diploma", "B.Tech", "M.Tech", "PhD", "Other"]
    backgrounds = ["CS", "IT", "ECE", "Mechanical", "Commerce", "Other"]
    programming_langs = ["Python", "Java", "C++", "JavaScript", "None"]
    tools = ["Power BI", "Excel", "Tableau", "TensorFlow", "PyTorch", "None"]
    interests = ["AI", "Web Development", "Cybersecurity", "Data Analysis", "Mobile", "DevOps"]
    soft_skills = ["Problem Solving", "Communication", "Leadership", "Teamwork", "Time Management"]
    preferred_times = ["Morning", "Night", "Anytime"]
    learning_styles = ["Visual", "Hands-on", "Reading", "Mixed"]
    preferred_domains = ["IT", "Govt Job", "Startup", "Research"]
    dream_roles = [
        "Software Developer", "Data Analyst", "AI/ML Engineer",
        "Cybersecurity Specialist", "DevOps Engineer", "Product Manager"
    ]

    job_roles = ["Software Developer", "Data Analyst", "AI/ML Engineer",
                 "Cybersecurity Specialist", "DevOps Engineer"]

    rows = []
    for i in range(1, n+1):
        age = random.randint(18, 40)
        education = random.choice(education_levels)
        background = random.choice(backgrounds)
        prog = random.choice(programming_langs)
        tool = random.choice(tools)
        interest = random.choice(interests)
        soft_skill = random.choice(soft_skills)
        study_hours = random.randint(0, 12)
        preferred_time = random.choice(preferred_times)
        learning_style = random.choice(learning_styles)
        preferred_domain = random.choice(preferred_domains)
        dream = random.choice(dream_roles)

        # Improved Label Rules 
        if interest == "AI" and prog == "Python" and tool in ["TensorFlow", "PyTorch"]:
            label = "AI/ML Engineer"
        elif interest == "Cybersecurity" or "Security" in soft_skill:
            label = "Cybersecurity Specialist"
        elif interest == "Web Development" or prog == "JavaScript":
            label = "Software Developer"
        elif interest == "DevOps" or tool in ["Docker", "Kubernetes"]:  # even if not in list now
            label = "DevOps Engineer"
        elif interest == "Data Analysis" and tool in ["Excel", "Power BI", "Tableau"]:
            label = "Data Analyst"
        else:
            
            label = job_roles[i % len(job_roles)]

        rows.append({
            "user_id": i,
            "age": age,
            "education_level": education,
            "background": background,
            "programming_languages": prog,
            "tools_known": tool,
            "interests": interest,
            "soft_skills": soft_skill,
            "study_hours_per_day": study_hours,
            "preferred_study_time": preferred_time,
            "learning_style": learning_style,
            "preferred_domain": preferred_domain,
            "dream_job_role": dream,
            "recommended_job_role": label
        })

    # Save dataset
    df = pd.DataFrame(rows)
    df.to_csv("intern1_it_jobs_dataset.csv", index=False)
    print(f"✅ Dataset saved as intern1_it_jobs_dataset.csv with {n} rows")
    print("Class distribution:\n", df["recommended_job_role"].value_counts())
    return df

if __name__ == "__main__":
    generate_dataset(100)

if __name__ == "__main__":
    df = generate_dataset(200)   
    print(df.head())
