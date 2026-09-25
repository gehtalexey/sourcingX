# Blind labelling task

You are an experienced technical recruiter. For each anonymised candidate profile, decide
what a careful recruiter should do for the given role. You see only the brief and the
profile. You are not told any other system's verdict.

Labels (exactly one per profile):
- `outreach` — worth contacting now: every must-have is shown or strongly implied, nothing
  in the profile contradicts a must-have, and no exclude rule applies.
- `needs verification` — nothing contradicts a must-have and no exclude rule clearly
  applies, but at least one must-have is not shown on the profile and would need a quick
  check (e.g. a call or a question in the first message).
- `reject` — the profile contradicts a must-have (wrong location, too little experience,
  wrong kind of role…) or an exclude rule clearly applies.

Judge strictly by the brief. Absence of evidence is not a contradiction.

## Output
Return ONLY a JSON array, one object per profile, in ID order:
`[{"id": "P01", "label": "outreach|needs verification|reject", "reason": "<one sentence, max 25 words>"}]`

## Brief: Dwelly (applies to profiles listed under "dwelly")
Role: Applied AI Engineer at Dwelly (AI-first UK lettings and property management
platform, $170M Series B), fully remote in the UK or Europe on UK hours, building
production agentic systems in TypeScript/Node and Python.
Must: At least 3 years of hands-on software engineering, backend (TypeScript/Node.js or
Python) / Has built and shipped AI or agentic systems that ran in production (tool use,
orchestration, structured outputs, evals, cost/latency/reliability), not just demos,
courses or side projects / Based in the UK or Europe / Degree in Computer Science or a
related technical field, or a strong engineering track record with another science degree.
Nice: Came from an AI-native company or a product startup / Technical founder history
(past founder now employed as an engineer) / Based in London, Spain, Portugal or Poland /
Uses coding agents in their own daily engineering work / PostgreSQL, LangGraph /
LangChain, LLM eval tooling.
Exclude: Data scientist, ML researcher, academic or research-heavy profile, or an "Applied
AI" title that is really classic ML / model training / Career mostly at IT consultancies,
outsourcing or software houses / Own company or solo freelancing is the only current job /
Currently at Dwelly.

## Brief: Owner (applies to profiles listed under "owner")
Role: Senior / Staff AI Agents Engineer at Owner (all-in-one platform for independent
restaurants), remote in US or Canada, building production LLM agents end to end.
Must: Minimum 5 years of software engineering / Has shipped production LLM or agentic
features (tool calling, agent loops), not just a simple LLM API call / Full-stack: works
across backend and frontend / UI / Based in the US or Canada / Bachelor's degree in CS or a
related STEM field / Worked at a strong startup backed by top-tier VCs.
Nice: LLM eval or observability tools (Braintrust, LangSmith, custom evals) / RAG /
retrieval systems in production / Restaurant tech, commerce or SMB tools background / Open
source work on agent frameworks or LLM tooling / Node.js / TypeScript / Big tech first,
then moved to a startup.
Exclude: Only research or theoretical AI, nothing shipped to production / Pure
backend-only or frontend-only specialist / Career mostly at consulting, outsourcing or
low-signal companies.


# Profiles


========== P01 — position: owner ==========
HEADLINE: Engineering at 11x.ai
LOCATION: San Francisco, United States

SUMMARY:
Serial builder

EXPERIENCE:
- Member of Technical Staff at 11x (2024-11 - present)
- Technical Cofounder, Fullstack Engineer - acquired by Teal HQ at Riva (2018-12 - 2020-09)
    - Product Mapping & Build: Collaborated with CEO to define, build, & evolve a web-based negotiations engine for job candidates using React + AWS + Serverless Framework, successfully navigating 4 major pivots in a fast-paced environment.
    - Robust Data Crawlers: Engineered robust web crawlers and ETL pipeline to extract compensation data from 12+ sources, feeding into an XGBoost model that powered data-driven negotiation strategies.
    - Product Team Buildup: Led the scaling of the product team by sourcing and managing 3 engineers, 1 data scientist, and 1 designer, enabling the launch of 2 major features that increased user retention by 25%.
    - Customer-Driven UI/UX: Redesigned our complex web interface for building negotiation responses, implementing intuitive user flows that resulted in an average 19% increase in annual compensation for customers.
- Cofounder, CTO at SteadyScript (2023-04 - 2024-11)
- Cofounder, CTO - acquired by OpenLoop at Imaging Panda (2020-10 - 2023-01)
    - Telehealth SaaS & API: Designed, built, and scaled a developer-friendly API with a responsive web interface for telehealth companies to send, track, and retrieve diagnostic imaging referrals from any medical imaging facility in the country.
    - Data Compilation & Enrichment: Constructed a comprehensive dataset of 16,000+ medical imaging facilities from dozens of fragmented data sources and developed unique heuristics to determine the range of diagnostics offered by each facility.
    - Patient-Facing Web App: Launched a patient-facing web app where patients can search affordable imaging facilities, forward their imaging referral, request an imaging study appointment, get SMS updates, and download their study results.
    - Full-Stack Architect: Architected full-stack solution using React (Typescript), Django, Redis, Docker, Nginx, and PostgreSQL, all deployed on AWS.
- RF Engineering Technician at Steren Electronics (2010-08 - 2011-09)
    - Characterized voice, video, and data communication solutions, such as cables, splitters, adapters, and amplifiers, over VHF and UHF bands
    - Lead a team of 10 contractors to repackage television satellite mounts to meet customer criteria
    - Designed serial DC isolator with <10 mdB insertion loss for DISH set top boxes
    - Coordinated Shanghai team to create, modify, and release electrical and mechanical design specifications
- Software Engineer, Backend at Facebook (2018-05 - 2018-09)
    - Metadata Service: Designed and implemented a dedicated metadata service to boost the reliability of API request throughput for mixed products.
    - Caching Mechanism: Employed a caching mechanism for read-heavy query patterns to cutback product latency by 66%.
    - Parallel Data Testing: Expanded the capability of internal tools with parallelized data integrity testing within our CI/CD, slashing CI/CD job time by half.
- Software Engineer (Intern) at Qualcomm (2017-06 - 2017-09)
    - Constructed vectorized mathematical instructions for Augmented/Virtual Reality SDK.
    - Applied fractional arithmetic and value lookup tables to offload computation to the mobile DSP processor.
    - Compartmentalized code so client can decide on their applicable speed-precision trade-off.
- Support Engineer at Qualcomm (2011-09 - 2015-11)
    - Owned and managed C# project integrated into system host application for debugging pixel-by-pixel calibration measurements 
    - Extensive use of Matlab for validation, test automation, and data analysis
    - Drafted, developed, and implemented CCA tracking system and application (C#) to interface Engineering Support Personnel to tracking spreadsheet
    - Identified and repaired complex CCA failures down to component level with a success rate > 90%
    - Modified and executed system power consumption script, analyzed data, and reported results for Mirasol wearable display
    - Enhanced system host application (C#) to interface from a single RS232-SPI device to multiple simultaneously connected devices
    - Collaborated with hardware and software teams to configure test environments for characterization
    - Developed and documented test procedures, board reworks, and system assemblies for Taiwan test team
- Cofounder, CTO - acquired by OpenLoop at Imaging Panda (2020-10 - 2023-01)
    - Telehealth SaaS & API: Designed, built, and scaled a developer-friendly API with a responsive web interface for telehealth companies to send, track, and retrieve diagnostic imaging referrals from any medical imaging facility in the country.
    - Data Compilation & Enrichment: Constructed a comprehensive dataset of 16,000+ medical imaging facilities from dozens of fragmented data sources and developed unique heuristics to determine the range of diagnostics offered by each facility.
    - Patient-Facing Web App: Launched a patient-facing web app where patients can search affordable imaging facilities, forward their imaging referral, request an imaging study appointment, get SMS updates, and download their study results.
    - Full-Stack Architect: Architected full-stack solution using React (Typescript), Django, Redis, Docker, Nginx, and PostgreSQL, all deployed on AWS.

EDUCATION:
- San Diego Mesa College — Associate's degree, Mathematics and Engineering Studies (2012-01 - 2016-01)
- Stanford University — Bachelor of Science - BS, Computer Science (2016-01 - 2019-01)

SKILLS:
Product Management, Analytics, Analog Circuits, System Testing, Amazon Web Services (AWS), Communication, React.js, Application Programming Interfaces (API), Web Development, Programming, Systems Design, TypeScript, Python (Programming Language), Databases, C#, Matlab, .NET, Failure Analysis, Debugging, C++, C, Soldering, Test Automation, Swift, GUI Designing, GUI development, GUI Testing, Electronics Repair, Data Analysis, Leadership, RF Troubleshooting, Engineering Support, Training, Research, Test Equipment, Microsoft Office, Electronics, Embedded Systems, Semiconductors, Engineering


========== P02 — position: dwelly ==========
HEADLINE: Senior Backend Engineer
LOCATION: London, United Kingdom

SUMMARY:
(none)

EXPERIENCE:
- Forward Deployed Engineer at Intercom (2026-03 - present)
- Senior software engineer at Tractable (2022-09 - 2026-03)
- Senior Backend engineer at Aircall (2021-09 - 2022-09)
- Senior Backend Engineer at Typeform (2021-03 - 2021-08)
- Advanced consultant/engineer at Altran (2017-11 - 2021-03)
    R&D aerospace and defense
- Software Engineer at GMV (2016-11 - 2017-11)
    Analysis and development of operational and validation solutions for satellite navigation systems.
- Software Engineer at Indra Group (2012-07 - 2016-11)
    Development of Business Intelligence solutions for Public sector.
- Intern at Indra Group (2011-09 - 2012-07)
    R&D Intern SW development

EDUCATION:
- Universidad Autónoma de Madrid — Computer Science & Engineering

SKILLS:
PHP, Laravel, Vue.js, JavaScript, J2EE Application Development, J2EE Web Services, Spring, Hibernate, Struts, JSP, Python, Django, .NET, C, C++, Docker, Microsoft SQL Server, MySQL, OLAP, Crystal Reports


========== P03 — position: owner ==========
HEADLINE: Software Engineer at Union.ai
LOCATION: Seattle, United States

SUMMARY:
(none)

EXPERIENCE:
- Software Engineer at Union.ai (2020-12 - present)
- Software Engineer at Google (2013-08 - 2016-06)
- Intern at ByteBite (2011-06 - 2011-08)
- Software Engineer Intern at Google (2012-05 - 2012-08)
- Tutor at Self Pace Center, UC Berkeley (2011-02 - 2011-05)
- Software Engineer at Lyft (2016-06 - 2020-12)
    Backend engineer working on Flyte: the open-source cloud native machine learning and data processing platform.
    Learn more at [link]

EDUCATION:
- University of California, Berkeley — Bachelor of Arts (B.A.), Computer Science (2009-01 - 2013-01)

SKILLS:
JavaScript, C++, Java, Python, Git, Databases, HTML, Distributed Systems, Ruby


========== P04 — position: dwelly ==========
HEADLINE: Product Software Engineer at Mistral AI
LOCATION: Paris, Île-de-France, France

SUMMARY:
Passionate about data and making things work, I have experiences on the whole data stack ranging from software engineering, data engineering, data science, AI engineering, ML Ops to data and cloud architecture.
I'm particularly interested in the healthtech, agritech and energy industries.

EXPERIENCE:
- Product Software Engineer at Mistral (2026-02 - present)
- Tech lead at OWKIN (2023-02 - 2026-01)
    Led a squad of 3 to 5 engineers to design, implement and maintain part of Owkin's in-house data governance and data science platform, helping data engineers to deliver AI-ready data, and data scientist and biomedical researchers to collaborate and find new therapeutic solutions in oncology.
    Specifically working on APIs and cloud infrastructure for data ingestion, storage, enrichment, processing, with the objective of making data easily accessible and actionable for scientists.
    Also contributed to K, Owkin's AI copilot for biomedical research (architecture, cloud deployment, agents as MCP servers)
- Machine Learning Engineer at Allianz Trade (2021-02 - 2023-02)
    As a Polyconseil consultant:
    - Design and implementation of an in-house self-service data science platform based on business and scientists needs in an AWS environment
    - Implementation of a feature store. Benchmark of existing solutions, design of an in-house solution, lead its implementation
    - Take part in designing and building other components of the data platform, such as a model registry, a data governance API and an API to provision compute instances
    - Promote develoment best practices
    Stack: AWS, Terraform, Python, Gitlab, Docker
- Data full-stack engineer at Polycea (2018-02 - 2023-02)
    As a data consultant, worked in various companies and industries (see other experiences) on the whole data stack, going from data engineering, to data science, to data-oriented software engineering and devops.
    Internally, in addition to these missions:
    - Improvement of in-house web scraping and ETL tools
    - Helped business development through some PoCs
    - Recruitment of new talents in the data team
- Data engineer - Software engineer at Allianz Trade (2019-04 - 2021-01)
    As a Polyconseil consultant:
    - Implementation and deployment of connectors between high value legacy on-premise databases and new cloud applications through an event bus. Enabled the new cloud, event-driven and microservices technical strategy
    - Factorization and development of a python package to improve the robustness and maintainbility of connectors
    Stack: AWS (DMS, RDS, ECS, Cloudwatch), Python, Terraform, PostgreSQL, Docker, Gitlab CI
- Data Scientist - Data Engineer at Sendinblue (is now Brevo) (2018-03 - 2019-03)
    As a Polyconseil consultant:
    - Implementation and deployment of a data processing pipeline to go from raw email logs (50M/day) to actionable datasets for data science
    - Design, training and deployment of a model to predict email openings (XGBoost)
    Stack: GCP, Python, Pyspark, MongoDB, Jenkins
- Data analyst at Acta les instituts techniques agricoles (2017-09 - 2018-02)
    Objectives: better inform farmers on crops epidemic dynamics, to lower the use of agrochemicals
    Missions: develop a tool to monitor and predict crops epidemics using statistical models (mixed models, RNN)
- Data analyst at Ariana Pharma (2017-03 - 2017-08)
    Objectives: Data analysis in personalized medicine for oncology, around an in-house therapeutic decision support tool that ranks treatments given a patient's genetic profile
    Missions: Improve the knowledge base using web scraping and text mining leveraging CNN on the literature. Analysis of the scoring algorithm
- R&D engineer at EDF (2016-03 - 2016-08)
    Objectives: Optimize offshore wind turbine structures to lower production costs
    Missions: numerical computing of offshore wind turbines structures using a frequency-based approach, taking into account constraints on modal frequencies, ULS and mechanical fatigue. Optimization of the structure using genetic algorithm
- Ingénieur mécanique at ECHY (2015-03 - 2015-07)
    Internship at Echy, a startup with an innovative lighting system that brings the comfort of natural light into buildings using fibre optic cables
    Missions: mechanical design of part of the new Echy product, in collaboration with designers
- Internal audit at Dassault Falcon Jet (2014-07 - 2014-08)
    Objectives: Internal audit of the manufacturing site of Little Rock, Arkansas
    Missions: Work with a small team (2 people). Preparation of the audit, interviews and data collection on site, analysis of the results

EDUCATION:
- Ecole polytechnique — Diplôme d'ingénieur polytechnicien
- MINES ParisTech — Diplôme d'ingénieur, Biotechnologie
- Imperial College London — Master of Science (MSc), Sustainable Energy Futures
- Lycée Janson de Sailly — Physique Sciences pour l'Ingénieur

SKILLS:
AI engineering, Software Engineering, Data Engineering, Cloud Computing, MLOps, Data Science, Python, Go, Terraform, Amazon Web Services, Google Cloud Platform, Analyses de donnés, Git, SQL


========== P05 — position: dwelly ==========
HEADLINE: Senior Software Engineer | Java, Kotlin, Cloud, Microservices
LOCATION: Sarajevo, Federation of Bosnia and Herzegovina, Bosnia and Herzegovina

SUMMARY:
As a Lead Engineer at Prewave, I work on building compliance-focused platform capabilities related to EUDR, enabling both customer and supplier engagement through scalable workflows and integrations. I work with Kotlin, Angular, GCP, Kubernetes, Pub/Sub, and microservice-based architectures, and I am responsible for driving technical delivery, implementing public APIs and third-party integrations, and ensuring high-quality execution across the team.Previously, I built backend and cloud-based solutions at Casumo and Klika, working across microservices, integrations, cloud migration, and technical leadership in fintech and other large-scale product environments. These roles helped me develop a strong foundation in designing reliable systems, leading engineering work, and delivering solutions aligned with business needs.I have a master's degree in Computer and Information Sciences from the Faculty of Electrical Engineering Sarajevo, and I am passionate about solving complex problems, learning new technologies, and delivering meaningful value through well-designed software systems.

EXPERIENCE:
- Lead Engineer at Prewave (2024-11 - present)
    I work as a Lead Engineer at Prewave on implementing EUDR regulation capabilities, enabling both customer and supplier engagement through the platform using Kotlin, Angular, GCP, Kubernetes, Pub/Sub, and microservice-based integrations.
    
    • Responsible for driving the technical delivery of compliance-related features covering both customer and supplier workflows within the platform.
    • Implemented public APIs for client integrations, enabling external systems to interact with Prewave compliance capabilities.
    • Built integrations with third-party APIs required for regulatory and compliance-related processes.
    • Delivered end-to-end flows that enabled customers and suppliers to operate in line with EUDR requirements.
    • Integrated platform capabilities with other departments through Pub/Sub-based asynchronous communication.
    • Led a team of engineers, ensuring on-time delivery, solution quality, and technical alignment across the implemented features.
    • Worked closely with stakeholders and product management on roadmap planning, requirements clarification, and prioritization.
- Senior Software Engineer at Prewave (2024-01 - 2024-10)
    I worked on Prewave's actions platform team, implementing supplier engagement through assessment surveys and actions using Kotlin, Angular, PostgreSQL, and GCP.
    
    • Owned both Kotlin backend services and Angular UI development for a key product area within the actions platform.
    • Implemented supplier assessment survey and action-related workflows, supporting engagement and follow-up processes within the platform.
    • Designed and evolved PostgreSQL schema and data model changes to support new product capabilities and improve maintainability.
    • Improved reliability, performance, and delivery speed of the feature area through continuous technical improvements across the stack.
    • Collaborated closely with product, design, and engineering teams to clarify requirements, shape solutions, and support rollout of new functionality.
- Senior Software Engineer (Contractor via ELC3) at Casumo (2022-12 - 2023-12)
    I worked as a Java and Kotlin software engineer on a microservice architecture for an iGaming client based in Malta.
    
    • Utilized advanced microservice architecture, including 150+ microservices, that was serving several client applications and following latest practices in cloud hosting and domain driven architecture. 
    • Responsible for building new payments flow for customers on several markets
    • Worked on improvements inside event sourced architecture that enabled smoother experience for the customers
    • Responsible for organizing data migration, with backward compatibility in mind, to improve the overall quality inside the architecture and deliver expected improvements identified by the business team
- Technical Lead at Klika (2021-10 - 2023-01)
    I worked on a Fintech project that is based on .NET Core technology stack for an USA based client
    
    • Building Greenfield project utilizing microservice architecture on Azure Kubernetes service. 
    • Responsible for backend development of microservices and architecture supporting them.
    • Created initial skeleton of order placement flow utilizing Behavior driven development and .NET Core best practices. 
    • Led a team of 7 engineers, assuring quality of the provided solutions, technical guidance to the team and constant communication with the clients regarding architecture and blockers.
    • Determined project requirements and developed work schedules for the team.
    • Delegated tasks and achieved daily, weekly, and monthly goals.
    • Liaised with team members, management, and clients to ensure projects are completed to standard.
    • Motivated staff and created a space where they can ask questions and voice their concerns.
- Technical Lead at Klika (2019-01 - 2021-10)
    I worked on Fintech project for Austrian bank that is based on Java and Spring Boot technology stacks.
    
    • Utilized advanced microservice architecture, including 150+ microservices, that was serving several client applications and following latest practices in cloud hosting and domain driven architecture. 
    • Responsible for organizing and helping cloud migration of our microservices to Azure Kubernetes Service, upgrading our technology stack to the latest version and removing technical debt in our solutions. 
    • Worked on integration of mobile applications and backend platform with other company departments and third party companies, adding asynchronous communication in the microservice architecture through Kafka framework. 
    • Had a role of Technical Lead to a backend team of engineers.
    • Responsible for assuring quality of provided solutions, technical guidance to the team and constant communication with solution architects about architecture development and improvements.
    • Ensured that items produced are in line with the technical designs and specifications of clients.
    • Interacted and exchanged ideas with project leads and other members of the team in a bid to arrive at good designs and solutions to the jobs at hand. 
    • Provided assistance to the technical director in creating and developing good project schedules.
- Software Engineer at Klika (2016-07 - 2019-07)
    I worked as a Backend engineer on IoT project building solutions and architecture for smart beds for USA based client.
    
    • Used .NET and Node.js technology stacks and Microsoft Azure for hosting our architecture. 
    • Built ETL processes for collection of various data related to our customers and their IoT devices.
    • Integrated third party applications and their data into our system.
    • Built an architecture for preparing and sending push notifications to mobile devices.
    • Developed advanced API's for exposing the data to mobile and web applications.
    • Used Node.js for developing data-intensive real-time applications that operate in distributed environments.
    • Focused on the complex and large software systems that make up the core systems for an organization.
- Student Partner at Microsoft (2015-04 - 2017-09)
- Intern at KING ICT (2014-11 - 2015-01)
    I worked as an Intern, which was part of "Moja Praksa" Internship program. The project was an ASP.NET application.

EDUCATION:
- Faculty of Electrical Engineering Sarajevo — Master's degree, Computer and Information Sciences, General
- Faculty of Electrical Engineering Sarajevo — Bachelor's degree, Computer and Information Sciences, General
- Faculty of Electrical Engineering Sarajevo — Bachelor's degree, Computer and Information Sciences, General
- Faculty of Electrical Engineering Sarajevo — Master's degree, Computer and Information Sciences, General

SKILLS:
Behavior-Driven Development (BDD), Azure Data Factory, Google Cloud Platform (GCP), DevOps, Kotlin, C#, Java, Node.js, Agile Methodologies, Cloud Computing, Object-Oriented Programming (OOP), JavaScript, ASP.NET MVC, Spring Boot, .NET Core, ASP.NET, Microsoft Azure, Microsoft SQL Server, Azure Kubernetes Service (AKS), .NET Framework, Docker, NoSQL, React.js, MySQL, Git, CSS, HTML, PostgreSQL, Azure Cosmos DB, Netflix OSS, Jenkins, Angular, Bootstrap, MongoDB, GraphQL, Spring Framework, Teamwork, Kubernetes, Apache Kafka, Microservices, TypeScript, SQL Azure, Cloud Services, Internet of Things (IoT), Back-End Web Development, REST APIs


========== P06 — position: owner ==========
HEADLINE: Building AI Agents | YC founder (W22)
LOCATION: New York, United States

SUMMARY:
(none)

EXPERIENCE:
- Founding MTS at Stealth (2026-04 - present)
- Co-Founder at OpCoder AI (2017-11 - 2018-07)
    Co-Founder of OpCoder AI - a technical revolution in healthcare billing and reimbursement fueled by natural language processing and artificial intelligence.
- Senior Software / Artificial Intelligence Engineer at The MITRE Corp (2018-07 - 2021-08)
    Software engineer specializing in computer vision, machine learning, and autonomous robotics.
- Lead Computer Vision Developer at Washington University in St. Louis (2016-08 - 2017-06)
    Led computer vision development for a bio-inspired autonomous vehicle called FlowBot. With a single monocular camera and OpenCV for Python, I implemented a real-time control algorithm using optical flow and machine learning to successfully navigate randomly generated obstacle fields. Finalist at the International Silk Road Robotics Innovations Competition at Xi’an Jiatong University in China (June 2017).
- Teaching Assistant, Data Structures and Algorithms at Washington University in St. Louis (2016-07 - 2016-12)
    Assist students through labs and coursework involving complexity analysis, lists, queues, stacks, priority queues, binary trees, red black trees, Dijkstra’s shortest path algorithm, and hashing. Helped create the website for the newly structured Data Structures and Algorithms course.
- Teaching Assistant, Computer Science 1 at Washington University in St. Louis (2015-08 - 2016-05)
    Helped students through various labs and studios using the Java language. Encouraged analytical thinking and problem solving abilities through direct interactions with students.
- Applied Aerodynamics Research Assistant at Department of Mechanical Engineering (2015-06 - 2016-01)
    Modeled wake contractions of a helicopter in hover and climb by use of Betz and Glauert distributions. Created figures on MATLAB and analyzed the figures to compare various aspects of the two distributions. Assisted in writing a paper covering background theory, derivations, and the processes used throughout my research that will be presented at European Rotorcraft Forum.
- Software Engineer at Boeing (2017-08 - 2018-01)
- Teaching Assistant, Rapid Prototyping and Project Development (ESE 205) at Washington University in St. Louis (2016-08 - 2017-05)
    Directly assist groups in a semester long systems and software engineering projects of their choice.
- Dynamic Stall Research Assistant at Applied Aerodynamics Lab (2016-06 - 2016-08)
    Helped develop a method to model the dynamic stall phenomenon in helicopters using a semi-empirical process. Utilized the Julia scripting language to find an accurate and fast method that may be used in flight simulations, aircraft embedded systems, and early development cycles of helicopter blades. Able to predict lift, moment, and drag with less than 10% error in a minute of computing time that may take weeks for a similar CFD program to compute.
- Student Associate at Washington University in St. Louis, First Year Center (2014-05 - 2015-05)
    Worked with one other Student Associate to serve as a resource for a group of 43 first year students. Programmed events around the city of St. Louis, advised students about academic and extracurricular activities, and managed our team’s budget of $250
- Cofounder & CTO at Uberduck (2021-08 - 2024-09)
    Built generative audio products used by millions of users and enterprises around the world.
- Member of Technical Staff at Hebbia.AI (2024-08 - 2026-06)

EDUCATION:
- Washington University in St. Louis — Master’s Degree, Computer Science (2017-01 - 2018-01)
- Washington University in St. Louis — Bachelor’s Degree, Mechanical Engineering, Computer Science, Robotics (2013-01 - 2017-01)

SKILLS:
Research, Teamwork, C++, OpenCV, Python, Java


========== P07 — position: dwelly ==========
HEADLINE: Full Stack Engineer at Synthesia
LOCATION: Croatia

SUMMARY:
Experienced Software Engineer with a demonstrated history of working on large projects like Skype and Microsoft Teams, web CRM and ERP systems and mobile application solutions. Excellent in Javascript with strong architectural and OOP skills. Passionate about learning new technologies like Rust, Web Workers, Web Assembly, functional programming and pushing the limits in modern web apps. Enjoys investigating project architectures to meet the goals of large and scalable applications.

EXPERIENCE:
- Full Stack Engineer at Synthesia (2021-09 - present)
- Software Engineer at Povio (2021-03 - 2021-09)
- Freelance Software Engineer at Povio (2020-11 - 2021-03)
- Software Engineer 2 at Microsoft (2019-03 - 2020-11)
    Working on Microsoft Teams desktop app to support crash reporting for multi windows in Electron app. Driving the project, organising the work and cooperating with other teams inside the Microsoft.
    
    Working in Skype App Core and Fundamentals team which provides app fundamentals support including low level Javascript logic (Database connections, Offline data syncing, Application startup, Performance optimisations, build scripts...).
    Developing modules for native platforms like Android and Electron to extend Javascript features with native code in order to fix performance issues and access platform API layer which is not possible from Javascript side.
- Software Engineer at Microsoft (2018-03 - 2019-03)
    Working on Skype application in Messaging team and developing frontend logic for multiple platforms using ReactXP framework. Development process is very data driven so we are paying a lot attention on user feedback through telemetry and AB testing.
    I'm covering the area of message synchronisation and performance, rendering of different types of messages (images, videos, links, emoticons...), Emoticon/GIF picker, Chat gallery, Share functionality...
- Technical Team Lead at netmedia (2016-12 - 2018-02)
    As a team lead for the web and mobile team I'm responsible for UX experience and technical solutions of NetMedia's products.
    Doing the project architecture, technology selection and development for big ERP and CRM custom systems in React, Angular and React Native.
- Senior Software Engineer at netmedia (2015-02 - 2016-11)
    Working as frontend developer mostly on business applications.
    Developing web applications in Angular/Angular 2 framework.
    
    Also developing hybrid mobile apps with Ionic 2 and Angular 2 framework for Android and iOS devices.
    Using latest web technologies: Typescript, Angular 2, Redux, RxJS Observables...
    Leading frontend department and managing agile and waterfall based projects
- Software Engineer at Studion (2012-03 - 2015-02)
    Implementing application modules for Refinery29 blog site and developing CMS system for content management and organization.
    Working on frontent code in javascript and writing Jasmin tests, and backend code in PHP.
    
    Programming digital catalogs and integration into client's web site with eCommerce compatibility in javascript. Worked on zmags framework updates and fixes.

EDUCATION:
- Faculty of Electrical Engineering, Mechanical Engineering and Naval Architecture, Split, Croatia — Master's degree, Electrical and Electronics Engineering
- Tehnička škola, Šibenik — Master's degree, Electrical and Electronics Engineering

SKILLS:
JavaScript, ReactJS, Recat Native, Front-end Development, Web Development, Web Applications, Object-Oriented Programming (OOP), Electron.js, GraphQL, Angular, AngularJS, Ionic, jQuery, CSS, HTML, Cascading Style Sheets (CSS), PHP, MySQL, TypeScript


========== P08 — position: dwelly ==========
HEADLINE: Speaker ¦ Founder ¦ Chair ¦ Investor ¦ Author ¦ 49⨯ Award Wins ¦ Private Equity ¦ Non Exec ¦ AI Engineer ¦ ESG Leader ¦ Working Class Hero
LOCATION: United Kingdom

SUMMARY:
Chair at Institute of Workplace 

Founder & Vice Chair at Pareto FM

Founder & Chair at Tomorrow Meets Today 

Past Chair at Emerging Workplace Leaders 

Private Equity Exits at NVM + Pictet

Visiting Lecturer at Oxford Brookes University 

International Speaker

Global Diversity Leader at Guardian Newspaper 

AI Prompt Engineer at Chat GPT

Author at The Power of Ten 

Author at Prickly Pals

EXPERIENCE:
- Chair - Non-Exec Director at IWFM | Institute of Workplace and Facilities Management (2025-08 - present)
- AI Prompt Engineer at OpenAI (2023-01 - present)
- Vice Chair - Non-Exec Director at Pareto Facilities Management Ltd (2022-08 - present)
- Founder & Chair at Tomorrow Meets Today (TMT) (2015-01 - present)
- #TMT25 at Tomorrow Meets Today (TMT) (2024-11 - 2025-10)
- Deputy Chair - Non-Exec Director at IWFM | Institute of Workplace and Facilities Management (2020-08 - 2025-07)
- #TMT24 at Tomorrow Meets Today (TMT) (2023-11 - 2024-10)
    [link]
- #TMT23 at Tomorrow Meets Today (TMT) (2022-11 - 2023-10)
    [link]
- Advisory Board Director at Rocking Ur Teens CIC (2020-01 - 2022-12)
- Founder & CEO at Pareto Facilities Management Ltd (2014-08 - 2022-07)
- IWFM Awards - Lead Judge at IWFM | Institute of Workplace and Facilities Management (2019-01 - 2020-12)
- Guest Speaker at Rocking Ur Teens CIC (2019-01 - 2019-12)
    [link]
- #TMT18 at Tomorrow Meets Today (TMT) (2018-11 - 2019-10)
    [link]
- IWFM Awards - Support Judge at IWFM | Institute of Workplace and Facilities Management (2014-01 - 2018-12)
- #TMT17 at Tomorrow Meets Today (TMT) (2017-11 - 2018-10)
    [link]
- #TMT16 at Tomorrow Meets Today (TMT) (2016-11 - 2017-10)
    [link]
- Governance Committee - Committee Member at IWFM | Institute of Workplace and Facilities Management (2015-03 - 2017-02)
- #TMT15 at Tomorrow Meets Today (TMT) (2015-11 - 2016-10)
    [link] 
    [link]
- Chair at Emerging Workplace Leaders (2013-01 - 2015-12)
- Lead Judge: Young Manager of the Year Award at Emerging Workplace Leaders (2011-12 - 2015-11)
- Rising FMs - Committee Member at IWFM | Institute of Workplace and Facilities Management (2011-05 - 2015-04)
    [link]
- Associate Director at Bilfinger UK (2012-08 - 2014-07)
    Accountable for service delivery across HSG division
- Committee Member at Emerging Workplace Leaders (2011-01 - 2012-12)
- Account Manager at Bilfinger UK (2009-08 - 2012-07)
    Operational responsibility for a number of accounts across London and the South
- Commercial Management Graduate at Bilfinger UK (2008-08 - 2009-07)
    Assisting with all commercial aspects of the business from costings to marketing

EDUCATION:
- University College London, U. of London — MSc, Facility and Environment Management (Distinction),
- University of Essex — BSc, Business Management (First Class Hons)

SKILLS:
Facilities Management, Commercial Management, Energy Management, Building Services, Contract Management, Operations Management, Management, HVAC, Procurement, Project Planning, FM, Corporate Real Estate, Account Management, Service Delivery, Facilities Operations, PFI, Building Management Systems, Change Management, Sustainability, Budgets, Building Management, Building Maintenance, Contract Negotiation, Property Management, Maintenance Management, CAFM, IOSH, Refurbishing, Energy Conservation, Bid Preparation, Private Sector, Energy Efficiency, Refurbishments, Facility Management (FM), Budgeting, Employee Training, Customer Service


========== P09 — position: owner ==========
HEADLINE: Staff AI Engineer
LOCATION: Sarasota, United States

SUMMARY:
A decade in software engineering, expertise in applied AI, and a strong creative edge. Published thought leadership content including a #1 Hacker News article on real-time AI game rendering.

EXPERIENCE:
- Senior Staff AI Engineer at Mechanical Orchard (2024-02 - present)
    Led development of generative AI COBOL comprehension and transpilation agents, improving speed, reliability, and accuracy by 10x.
- Senior Software Engineer at Pivotal Tracker (2015-08 - 2017-10)
    Worked across the stack on Pivotal Tracker adding features on both ends of an event-sourced, real-time API used by hundreds of thousands of users.
    Front-end lead on the web app.
    Also lead development of internal tools and prototypes for CI/build visualizations.
- Senior Software Engineer at Geometer LLC (2020-07 - 2021-09)
    Incubator focused on many projects.  I worked on multiple game development projects including:
    • A 3D, social, narrative rich game where I focused on NPC personality modeling based on backstory, goals, and a multilayered emotion graphs
    • Applying procedural narrative generation research from Chris Martens whitepapers on "Linear Logic Programming for Narrative Generation"
    • A slither.io clone to explore real-time scaling patterns with Elixir
    • Lead on a distributed, highly scalable Entity Component System (ECS) server, store and syncing protocol in Elixir, with a large scale HTML multiplayer game to test it out.
- Interim CTO at Oneirocom (2022-05 - 2022-12)
    Whirlwind side-hustle alongside some of the "AI Dungeon" developers to bootstrap and fund a start-up in the generative AI space, specifically building a generative AI narrative engine and tooling stack for games, NPCs and immersive experiences.
    
    My roles included building demos and proofs-of-concepts, collaborating with ML researchers to combine traditional AI and game dev techniques with LLMs and other generative approaches, and networking with industry specialists.
- Senior Software Engineering at Kohort (2021-09 - 2022-10)
    Spin-out startup from Geometer.  Elixir tech stack building video communication and remote collaboration platform.
- Javascript Game Developer at Two Bit Circus (2012-10 - 2013-03)
    Built HTML5 based educational games.
- Senior Software Engineer at Moneyhub Enterprise (2013-11 - 2015-07)
    Full stack developer hired on visa to help build an ambitious fin-tech product at a fast-growing startup.
- Lead Software Engineer at Mechanical Orchard (2022-11 - 2024-02)
    Let zero-to-one R&D initiatives in AI assisted mainframe comprehension and modernization, securing $24M Series A funding round.
- Senior Software Engineer at Adobe (2017-10 - 2020-07)
    Part of a small, experimental team, building a real-time, collaborative photography app.  Also designed, built, and evangelized an innovative synthetic users system for load generation and automated testing across multiple Adobe teams and projects.
- Software Engineer at BLT (2013-03 - 2013-05)
    Built interactive media websites for Hollywood studios, including a project for True Detective.

EDUCATION:
- University of Colorado Boulder — Bachelor of Fine Arts - BFA, Film Studies and Production (2000-01 - 2000-01)
- Coursera — Course Certificate, Generative AI with Large Language Models (2023-08 - 2023-09)

SKILLS:
RAG, Technology Roadmapping, Product Road Mapping, Large Language Models (LLM), Generative AI, Prompt Engineering, TypeScript, Python (Programming Language), Communication, React.js, Elixir, Google Cloud Platform (GCP), Game Design, Game Development, Amazon Web Services (AWS), Unity, Artificial Intelligence (AI), OpenAI Products, Machine Learning, Agile Methodologies


========== P10 — position: dwelly ==========
HEADLINE: Engineer at Lovable
LOCATION: Stockholm, Stockholm County, Sweden

SUMMARY:
Engineer who likes to lead, bridges the gap between product and tech, and tends to be the one people ask to explain how something works.

EXPERIENCE:
- Member of Technical Staff at Lovable (2026-05 - present)
- Senior Backend Engineer at Instabee (2023-05 - 2026-05)
    3 years on route optimization. Go, distributed systems. End-to-end ownership of the messy cross-team projects
- Software Engineering Consultant at Mpya Digital (2022-03 - 2023-05)
    Backend and full-stack consulting across Stockholm tech.
- Machine learning engineer at Hedvig (2021-08 - 2022-01)
    Built and trained fraud detection models for insurance claims.
- Software Engineer at Hedvig (2021-05 - 2022-01)
    Backend engineer during Hedvig's scale-up.
- Software Engineer at Airpelago (2020-05 - 2021-01)
    Early engineer at Airpelago. Now the Nordic market leader in drone-based power line inspection
- Math tutor at Linköping University (2019-10 - 2020-01)
    Led the classes that complemented lectures in Calculus for engineering students.
- Travel Guide at STS Alpresor (2012-11 - 2013-04)

EDUCATION:
- Linköping University — Master's degree, Computer Science and software engineering
- University of the Sunshine Coast — Bachelor's degree, International Business
- University of the Sunshine Coast — First year of a bachelor's, International Business
- Linköping University — Master's degree, Computer Science with GPA of 4.9

SKILLS:
TypeScript, Amazon Web Services (AWS), Google Cloud Platform (GCP), Go (Programming Language), Distributed Systems, Cross-functional Team Leadership, Project Management, SQL, NoSQL, Python (Programming Language), Machine Learning, Artificial Intelligence (AI), C++, Java, Volunteering, Teaching


========== P11 — position: dwelly ==========
HEADLINE: Senior Software Engineer at Cognigy
LOCATION: Berlin, Berlin, Germany

SUMMARY:
Computer engineer with 8 years of experience, I possess a strong background in full-stack…

EXPERIENCE:
- Senior Software Engineer at NiCE Cognigy (2025-02 - present)
- Senior Software Engineer at Grover (2022-09 - 2025-01)
    Lead architecture, prioritization and implementation of cross-domain projects that impact the team's operational and infrastructural reliability, customer experience, manual resolution of issues, as well as GDPR compliance.
    
    Regular on-call responsibilities, setting up key metric dashboards, monitors, and ensuring smooth operation of our services 24/7 even during key seasonal events for retail such as Black Friday and Cyber Monday weeks.
- Software Engineer at Grover (2021-06 - 2022-09)
    Architected and implemented event-driven data aggregation microservice for customer order insights, integration of UPS tracking services, enabling US market expansion in 2021, mentored new team members, led 5-engineer team in hackathon, managing tasks and presenting to judges, conducted technical interviews for engineering candidates.
- Full-stack Developer at Fraugster (2019-05 - 2021-05)
    Contribute developing the customer facing application and underlying microservices allowing customers to configure the fraud prevention tools to suit their business's needs, as well as continuously extending such customization options.
    
    Lead the development of an service that allowed our sales team and customer relationship managers to create sandbox environments on-demand to show and onboard the features and customization options we offered to customers who wanted to manage and tune our services to better fit their needs.
- Software Engineer at Forto (2018-01 - 2019-04)
    Fullstack development of FreightHub's micro-services based freight forwarding solution, pragmatic engineering solutions for a key customer integration involving the use of EDIFACT formatted communications, which would provide our customer with an EDIFACT-compatible API that could be consumed from within their existing systems, providing them seamless access to our services.
- Semi-senior Developer at Hexacta (2017-11 - 2018-01)
    Development and implementation of features and process optimisation for a project management solution for the Buenos Aires city government.
- Full-stack Developer at Humbee Partners (2017-08 - 2017-11)
    Android client and server development and maintenance, web-socket communication implementation in both client and server, GraphQL implementation on both the client and server.
- Backend Developer at Humbee Partners (2016-10 - 2017-07)
    Application usage and traffic reports; process development and maintenance for web and mobile application platforms, deployment automation or continuous delivery, regular experimentation and implementation of new technologies and development tools.

EDUCATION:
- Universidad Rafael Urdaneta — Engineer’s Degree, Computer Engineering

SKILLS:
Datadog, Apache Kafka, NestJS, Terraform, TypeScript, Node.js, Go (Programming Language), Scrum, JavaScript, Angular 2, PostgreSQL, Docker Products, React.js, MongoDB, GraphQL, Android, Java, Kotlin, SQL, MySQL, Git, Kubernetes


========== P12 — position: owner ==========
HEADLINE: AI Agent Orchestrator
LOCATION: United States

SUMMARY:
Experienced tech lead and manager in distributed systems, learning LLM.

EXPERIENCE:
- Member of Technical Staff at OpenAI (2026-06 - present)
    Cooperative AI. 
    Teach AI to work with human.
- Software Engineer at Uber (2015-08 - 2017-08)
    Uber business foundation platform.
    Early infra team member.
- Software Engineer at Google (2017-08 - 2022-01)
    TL leading A/B testing of Google Play Store.
- Software Engineer at OpenText (2013-06 - 2015-07)
- Software Engineer at Meta (2023-11 - 2026-05)
    TL of AI Annotation Data and Eval Platform. Support TBD and previous Llama teams for post-training by providing high quality datasets.
    
    Led consolidation and unification effort on building platforms that manage all Meta Annotation Data and Evals. AI, human experts, Vendor, in-house. 
    
    Uber TL in driving  AI for automation, evaluation and human in the loop application to transition internal operations to AI-native workflow.
- Engineering Leader at Sibros (2022-01 - 2023-06)
    TLM running Sibros a global cloud data team.
- Principal Engineer at Stealth AI Startup (2023-06 - 2023-11)
    Build queryless data warehouse solutions leveraging LLMs.

EDUCATION:
- University of Southern California — Computer Science

SKILLS:
Trino, Data Annotation, Agentic LLM Applications, LangChain, Large Language Models (LLM), Business Intelligence (BI), Android Development, Big Data, Reliability Engineering, Experimentation Frameworks, Software Release Management, Apache ZooKeeper, Data Analytics, Software as a Service (SaaS), Experimentation, Android, Quality of Service (QoS), Distributed Caching, Microservices, Load Testing, Data Lake, Google Cloud Dataflow, Google Kubernetes Engine (GKE), Team Leadership, A/B Testing, ClickHouse, Data Lakes, Amazon Web Services (AWS), Google Cloud Platform (GCP), Apache Druid, Apache Flink, Apache Spark, Java, MongoDB, Cassandra, Apache Kafka, Docker, ElasticSearch, Machine Learning


========== P13 — position: dwelly ==========
HEADLINE: Senior Software Engineer bei SoundHound Inc.
LOCATION: Berlin, Berlin, Germany

SUMMARY:
(none)

EXPERIENCE:
- Senior Software Engineer at SoundHound AI (2021-11 - present)
    Lead Developer Automotive AI Customization
- Software-Entwickler at PSI Software (2012-07 - 2021-10)
    Entwickler im Bereich sicherheitskritischer Software zur Steuerung und Überwachung von Pipelines. 
    Dazu zählt unter anderem die Weiterentwicklung eines numerischen Simulationskerns zur physikalischen Echtzeitsimulation von Pipelines, Entwurf und Umsetzung von Algorithmen zur Leckerkennung und Leckortung, Daten-Analyse bei Inbetriebnahmen, Betreuung von SIL-Zertifizierungen der Software, Rufbereitschaft im Rahmen einer 24/7-Verfügbarkeit, Design und Umsetzung einer Container-basierten Microservice-Architektur mit Message-Broker-Kommunikation.

EDUCATION:
- Technische Universität Berlin — Diplom, Mathematik
- Technische Universität Berlin — Diplom, Mathematik

SKILLS:
Artificial Intelligence (AI), Prompt Engineering, Spring Boot, Quarkus, Git, Englisch, React.js, Node.js, Natural Language Understanding, TypeScript, C++, Programmieren, Programmierschnittstellen, Numerische Analyse, Mathematische Modelle, Hydraulik, Simulationen, Pipelines, Java, Python (Programmiersprache)


========== P14 — position: owner ==========
HEADLINE: Lead Engineer
LOCATION: Chicago, United States

SUMMARY:
(none)

EXPERIENCE:
- Lead Engineer at Ahold Delhaize USA (2026-06 - present)
- Lead Engineer at EGEN Solutions, Inc. (2022-10 - present)
- Lead Engineer at Pryon (2024-12 - 2026-06)
- Lead Engineer at Pyxos (2023-07 - 2024-12)
- Lead Engineer at DriveTime (2022-10 - 2023-06)
- Senior Software Engineer Team Lead at EGEN Solutions, Inc. (2021-06 - 2022-10)
- Senior Software Engineer Team Lead at DriveTime (2021-06 - 2022-10)
- Software Engineer at EGEN Solutions, Inc. (2018-10 - 2021-06)
- Software Engineer at Tempus AI (2018-10 - 2021-06)
- Cyber Risk Analyst at Deloitte (2014-08 - 2016-04)

EDUCATION:
- Ira A. Fulton Schools of Engineering at Arizona State University — Master of Science - MS, Computer Engineering (2016-08 - 2018-05)
- CVR College of Engineering, Hyderabad — Bachelor of Technology - BTech, Electrical and Electronics Engineering (2010-10 - 2014-04)

SKILLS:
FastAPI, Next.js, Kubernetes, Google Kubernetes Engine (GKE), Artificial Intelligence (AI), Large Language Models (LLM), Spring Boot, PostgreSQL, ArcSight, Microsoft Azure, IndexedDB, MySQL, Google Cloud Platform (GCP), .NET Framework, Apollo GraphQL, Microsoft SQL Server, Python, TypeScript, Ionic Framework, Machine Learning


========== P15 — position: dwelly ==========
HEADLINE: Senior Software Engineer at Mistral AI
LOCATION: Paris, Île-de-France, France

SUMMARY:
I’m a senior full-stack engineer and product builder with a strong systems mindset, experienced in designing and shipping end-to-end web products in fast-moving environments. I work across backend architecture, APIs, data flows, and user-facing interfaces, with a focus on correctness, performance, and long-term maintainability.

I stay hands-on while operating at a high level of abstraction, designing the structure and constraints of systems and using modern AI-assisted development tools to accelerate iteration without compromising engineering quality. I thrive in ambiguous problem spaces and enjoy turning early ideas into reliable, production-ready software.

EXPERIENCE:
- Senior Software Engineer at Mistral (2026-08 - present)
    Working to bring Vibe CLI to the next level
- Technical Lead at Gelato (2026-01 - 2026-08)
    Leading the technical delivery of one of Gelato's most strategic initiatives: enabling job-based orders within the platform. Previously, customers could generate estimates through a standalone module, but those estimates could not be converted into production-ready orders. This initiative established the missing link between estimating and the end-to-end order lifecycle, unlocking a core capability for enterprise customers.
    
    As the Technical Lead, I drove the initiative from architecture to delivery, aligning multiple engineering teams and defining the technical direction across a highly distributed platform.
    
    My main duties included:
    - Leading the architecture and technical execution of the job-based order initiative, from discovery through production delivery.
    - Designing and delivering the integration between the estimating platform and Gelato Connect's core services, including Product Platform, Order Service, Workflow, Logistics, Procurement, and Imposition, enabling end-to-end order creation from estimates.
    - Personally owning the implementation of several critical service integrations while ensuring scalability, reliability, and maintainability.
    - Driving cross-team alignment and technical decision-making across multiple engineering teams, resolving dependencies and maintaining delivery momentum.
    - Working closely with Product Managers, Engineering Managers, and Tech Leads to shape requirements, remove blockers, and deliver a strategically important platform capability.
    - Leading architectural discussions, reviewing designs, mentoring engineers, and contributing hands-on through implementation and code reviews to maintain a high engineering quality bar.
- Technical Lead at Gelato (2025-02 - 2026-03)
    Leading the development of a new layout automation engine for printed products (imposition engine), enabling dynamic, real-time design and efficient production workflows. Working across the stack to build a live template editor and a scalable, high-performance backend used by our print partners worldwide. Also overseeing maintenance and gradual retirement of our legacy layout system to ensure a smooth platform transition.
    
    My main duties include:
     - Sprint planning, backlog grooming, and participating in product discovery to shape upcoming work.
     - Leading architectural decisions across the stack — from backend and frontend design to UX — ensuring scalable, maintainable, and user-friendly solutions.
     - Providing both formal and informal mentoring through regular one-on-ones, code reviews, pair programming, and enhancing leadership skills via a dedicated leadership development program.
     - Actively contributing to coding by developing new features, fixing bugs, and optimizing performance. 
     - Collaborating regularly with product managers, designers, and other engineering teams to align technical direction, coordinate projects, and help unblock obstacles for smooth cross-team delivery.
- Technical Lead at Gelato (2021-11 - 2025-02)
    Project: Leading the development of a 2D design editor, allowing customers to add content such as image and text and manipulate them, also responsible for other frontend deliverables. Leading the development of the backend service creating mockups and files to be printed on products.
    
    Work: 
    - Providing technical leadership and mentorship to a small team while remaining hands-on on core features and architecture
    - Balancing business and technical requirements
    - Taking care of the development processes
    - Managing cross team requirements and communication
    - Developing new features
    - Pushing for quality improvement of our deliverable: drastically reducing client side errors on Sentry, integrating core web vitals monitoring
    - Improving our development experience: faster reliable tests, offline development sandbox
    
    Technologies:
    - frontend: React.js with Typescript, Redux, Fabric.js, WebGL, and Playwright.
    - backend: Python, OpenCV, Nodejs, AWS.
- Senior Frontend Engineer at Gelato (2021-01 - 2021-11)
    Project: Developing a 2D design editor, allowing customers to add content such as image and text and manipulate them. 
    
    Work:
    - Developed new features: elements snapping with each other, support for new content like calendar grid
    - Improved the current solution: created a small abstraction layer allowing to use Fabric.js as a first class citizen of React.js, benefitting from React.js component lifecycle management
    - Continuously pushed to improve solutions, as much on the technical side as the UI/UX side
    - Maintained the system in production
    - Mentored junior/mid engineers and helped with the recruitment
    
    Technologies: React.js with Typescript, Redux, Fabric.js & WebGL.
- Software Engineer at Indeed (2020-03 - 2021-01)
    Project: Developing a Candidate recommendation engine: for a given job, return a list of most likely matching candidates using ML + search. Focusing on internationalizing this engine.
    
    Work: 
    - Improving the current data pipeline to allow more data source and different languages/countries
    - Improving the Elasticsearch pipeline to allow multiple languages
    - Working on the ML model to adapt the system to handle multiple languages
    
    Technologies: Java, Apache Hadoop & Spark, AWS EMR, Elasticsearch.
- Software Engineer at Indeed (2019-03 - 2020-02)
    Project: Developing a messaging application for employers to communicate with candidates through the Indeed app.
    
    Work:
    - Drove architecture decisions and brought new technologies
    - Implemented a big part of the application
    - Mentored new engineers
    - Received an engineering ownership award
    
    Technologies: ReactJS, Typescript, Java.
- Lead Engineer / Full stack at  (2018-03 - 2019-01)
    Fullstack re-development of the application:
    - on the server: Typescript with Node.js, PostgresSQL, Serverless, AWS services (Elastic Beanstalk, Cognito, SQS, Lambda), Elasticsearch, Redis and Docker.
    - on the client: Typescript with React.js, custom MobX like library to easily integrate with GraphQL.
    
    I lead the development of the front and the back, managing 2 engineers. I took care of most of the infrastructure, deployment and continuous integration, many of the backend features such as upload & thumbnails generation, download, client live updates, and i developed most of the client (desktop version only so far).
- Lead Front-end Engineer / Full stack at SENSU K.K. (2016-01 - 2018-03)
    - Lead the front-end development of 3 differents clients (desktop, mobile and chrome extension) over 2 years with a team varying between 1 and 4 engineers.
    - I Had to take over the back-end (in Node.js), which includes bug fixing, developing new features and doing data migrations.
    - Created an performant interface with React.js which allows multiple users to work on the same touchscreen client. I wrote a custom library to handle any kind of user interactions easily and a polyfill to support the PointerEvent API on Safari ([link]
    - Designed and implemented the communication layer on both the back-end the front-end (using  WebSockets and REST APIs), which allow live updates, and optimistic updates.
    - Created a fully animated drag and drop system.
    - Set up continuous integration and deployment using CircleCI 2, Karma/Mocha and Jest.
    - Managed all the clients in a mono repo project to easily shared components. Setup the build system using Webpack.
    - Migrated the stylesheets to follow the BEM methodology in combination with LESS.
    - Switched the entire codebase to Typescript early on.
    - Migrated our client data management to MobX.
    - Designed and implemented on both the front and the back the sharing feature of our file system.
    - Suggested countless ideas and implemented them.
    
    Technologies : Typescript, React.js, MobX, LESS, Webpack, Websockets, Node.js, AWS.
- Frontend Engineer at TableCheck (2015-07 - 2016-01)
    Development:
    Table Solution ([link] a web-app and tablet app. Technologies: Ember.js (CoffeeScript / Emblem / SASS) and Cordova
    I mainly refactored an application written with an old Ember version, updated it in order to use newer version of our dependencies.
    Started the development of Table Check ([link] a web-app and a mobile app. Technologies: ReactJS (Javascript / LESS) and Cordova.
- Game Developer at Wizcorp (2013-09 - 2015-07)
    Application architecture, UX/UI development, mainly using Node.j, Cordova, Javascript, HTML5 and CSS3. All the DOM and style manipulations was made through javascript directly. All of the apps are single-page apps.
    - NBA2K: I joined an ongoing project, which went in production one month after. My job was to fix, improve and add new features to a mobile game.
    - LoveTsuri: I was part of a team developing from scratch the desktop and mobile browser clients of a game.
    - Dofus ([link] The project was about porting of a successful MMORPG game (Dofus - [link] in Flash to web technologies in order to use it on tablets. I was then lead UX/UI engineer, my job was to migrate the application logic from another codebase, in a different language, to the browser. Find elegant UX solution to use the desktop features in a touch environment. I was in charge of the development of many parts part of the application, mostly focusing on the UI: multi windows system, interaction handler to control various behavior easily, pointer event polyfill (to get hover hooks on touch based system).
- Research Development Software Developer at Silicon Studio (2013-02 - 2013-08)
    Development of a texture handling library (compression, mipmaps generation, atlas creation, ...) and an audio compression library for the asset pipeline of a new 3D game engine in C# ([link] 
    Porting of this engine on iOS (using Mono).
- Software Engineer at Atos (2012-06 - 2012-08)
    Enhancement of a storage service for Orange Business Service
- Software Engineer at TRAME (2011-06 - 2011-08)
    Creation of an online estimation software in PHP.

EDUCATION:
- Letterkenny Institute of Technology — Bachelor, Computer Games Development, Computer Science
- Letterkenny Institute of Technology — Bachelor, Computer Games Development, Computer Science
- INSA Rennes — Engineer's degree, Computer Software Engineering

SKILLS:
(none)


========== P16 — position: dwelly ==========
HEADLINE: Applied AI Engineer at MistralAI
LOCATION: Palaiseau, Île-de-France, France

SUMMARY:
4rd year student at Mines Paris PSL after 3 years at Polytechnique
Specialized in Applied Mathematic and Data. 
Main subject : Data Science and AI

EXPERIENCE:
- Applied AI engineer at Mistral (2025-04 - present)
- Machine Learning Engineer at Apple (2024-07 - 2025-04)
- Machine Learning Engineer at Apple (2023-12 - 2024-07)
- ML Engineer - Researcher- Software Developer at Datakalab (2023-03 - 2023-12)
    Development of neural network compression methods.
    Research on Stable Diffusion and quantization of transformers. Sofware development
    in python. Framework: Torch, TensorFlow, Onnx, TensorRT
- Data Scientist - France Field Medical Specialist - Health Strategy, Medical relation at Amplifon (2022-03 - 2022-07)
    Use of Artificial Intelligence techniques to identify different patient profiles and propose appropriate hearing aids.
    Objectifs : identify the main determinants of prosthetic outcome, and analyse their impact.
- Software Development Engineer at SNCF Réseau (2021-06 - 2021-08)
    As part of the development of the OSRD (open source railway designer) simulation tool, development of front/back algorithms, in java and react.js, to enable the implementation of mobile blocks in the simulation.
    
    Discovery of the workings, techniques, systems of relationship, organisation, innovation and management within a department or a company.
- Colleur de physique at Lycée Marcelin Berthelot (2020-09 - 2021-06)
    Interrogateur de physique en classe préparatoire
    1 h en MPSI et 1 h en PSI* par semaine
- Intervenant X-Projet at Hummatch Smart Hiring (2020-11 - 2021-03)
    Développement et design d'un site Web en node.js. 
    Objectif du site : proposer des jeux pour permettre aux entreprises de mieux recruter en offrant la possibilité de trier les participants en fonction des résultats obtenus et du profil recherché.
    
    Première expérience professionnelle, découverte des relations avec un client. 
    Développement entier du site basé sur node.js, et Django
    Travail en équipe de 3 personnes
- Stage en gendarmerie at Gendarmerie Nationale (2019-10 - 2020-03)
    Détachement opérationnel de 4 mois au PSIG de Besançon, précédé de deux mois de formation à l'école des officiers de gendarmerie.
    J'ai servi dans l'unité d'intervention en tant qu'adjoint a l'officier responsable du PSIG.

EDUCATION:
- École Polytechnique — Diplôme d'ingénieur, Ingénierie
- Mines Paris - PSL — Diplôme d'ingénieur, Mathématiques appliquées
- Lycée Marcelin Berthelot — Sciences physiques
- Lycée Marcelin Berthelot — Baccalauréat, Sciences physiques

SKILLS:
Core ML, Generative AI, IA responsable, Data Engineering, Python (Programming Language), Machine Learning, TensorFlow, Onnx, Artificial Intelligence (AI), Software Development, Intelligence artificielle (IA), Recherche clinique, PyTorch, XGBoost, Science des données, Pandas (logiciel), Ingénierie, Management, Java, Python


========== P17 — position: owner ==========
HEADLINE: Software Engineer at Warp, Founder at Hack The 6ix. Ex-YC founder.
LOCATION: Toronto, Canada

SUMMARY:
I'm currently a software engineer at Warp.dev, an Agentic Development Environment. We're an AI-first terminal designed for an agent-first development workflow.

I was a founding engineer at Vellum.ai, an ecosystem of tools that helps everyone build production-ready LLM applications. 

I spent 4 years as the co-founder of Venue, a video platform that helps remote-first companies make their all-hands more fun. We are YC W23 alumni and raised 5MM from Accel.

I was previously the founder of CodeMode, a small development agency specializing in building MVPs. Our clients ranged from individual entrepreneurs to mid-sized corporations, and grossed >$500k of revenue in less than 2 years.

I founded Hack The 6ix, a non-profit dedicated to growing young tech entrepreneurs by organizing student hackathons. Our hackathons have been attended by thousands of bright young minds, and that number is growing every year.

I'm also the co-founder of Beatcamp, an e-commerce platform that facilitates the licensing of instrumental hip-hop music between artists and producers. We received a $60k grant from the OCE SmartStart program, and made over $50k in sales.

I previously worked as a software engineer at Wealthsimple, a quickly growing startup disrupting the finance industry by making great financial services easily accessible to everyone.

EXPERIENCE:
- Founder, Board Chair at Hack the 6ix (2022-11 - present)
    Hack The 6ix is a non-profit that helps students pursue entrepreneurship through events such as hackathons, workshops, and social events.
    
    We've been organizing hackathons and workshops for 5 years, with an alumni network of over 100 organizers and volunteers. Our live and virtual events have brought together and educated 1,000s of students worldwide.
    
    I'm currently serving on the Board of Directors, and providing mentorship to the team as needed.
- Software Engineer at Warp (2025-08 - present)
    Warp is an Agentic Development Environment - an environment purpose-built for an agent-first software development workflow.
- Co-Founder, CTO at Beatcamp (2016-08 - 2018-12)
    Beatcamp is an e-commerce platform that helps rappers make better music and helps producers make more money.
    
    - Developed the Beatcamp web platform which serves 1,000+ producers, hosts 8,000+ beats, and handled $50,000+ in sales
    - Developed the Beatcamp mobile app, enabling producers to record on the go
    - Responsible for all technology, including development, devops, and security
    - Managed a team of 3 developers
- Founder, Board Member at Hack the 6ix (2016-03 - 2022-11)
- Founding Engineer & Eng Lead at Vellum (2023-08 - 2025-05)
    Vellum is a suite of tools that help technical and non-technical persons ship and maintain LLM applications. I joined as a founding engineer shortly after YC/Seed.
    
    In my time here, I worked on tooling for LLM developers covering Prompt Engineering, Agents, Evaluations, Deployment & Monitoring.
    
    I owned the Evals product vertical, and was ultimately responsible for everything that shipped. I lead a cross-functional team of 5.
    
    My projects spanned the entire development process, covering research, planning, resourcing, development, deployment, and maintenance.
- Co-Founder, Advisor at Venue (2023-06 - 2024-06)
- Software Engineering Intern at Shopify (2014-09 - 2015-04)
    •	Created real time data analysis tools
    •	Developed a customized query language for internal use
    •	Designed and implemented UI for data analytics
    •	Integrated timezones supporting consistent data worldwide
    •	Developed features for Facebook Buy Buttons
- Founder, Hackathon Chair at Hack the 6ix (2015-07 - 2016-03)
- Software Engineer at Wealthsimple (2016-05 - 2018-01)
    Projects:
    
    - High Interest Savings Accounts
    - Earn Rewards 2.0
    - Gifting investments through Wealthsimple
    - Account activity calculation/aggregation overhaul
    - Fax transfer automation
    - Return rate calculation (TWR, MWR, Simple)
    - Account opening automation
    - Withdrawals automation (30,000 withdrawals processed to date)
    - Tax withholding for withdrawals
    - Transfer model overhaul
    - Document upload/download microservice
    - Instant bank verification
- Co-Founder & Eng Lead at Venue (2020-04 - 2023-05)
    Venue helps the world's best remote-first companies like Shopify, PwC, and more run awesome online meetings and webinars. We're an alumni from YC W22, and raised our seed round from Accel.
    
    As a co-founder and engineering manager at Venue.live, I recruited and managed a team of 7 engineers. I fostered a remote-first culture with effective communication and collaboration, and provided guidance and mentorship to help my team grow to become more thoughtful and impactful engineers.
    
    As an engineer, I built most of the v1 of our core product, and acted as the lead architect for major features. I built systems and standards that enabled other engineers to be more productive and effective. I held the team to high technical standards, providing mentorship and feedback where appropriate to achieve and maintain those standards.
    
    Some architecture and engineering accomplishments I'm proud of:
    - Increased meeting capacity from 100 to 10,000
    - Improved random connection matching performance by 4,517%
    - Designed a stable RTC abstraction that allowed developers to seamlessly switch between multiple unstable RTC providers
    - Designed a "field-of-view" subscription system that gave attendees the illusion of being connected to everyone while only being connected to a small subset at any given time
    - Created most of Venue V1 :)
    
    I was hands on with all parts of the product, including:
    - Frontend (React)
    - Backend (Ruby, Node)
    - Infrastructure (AWS, Google Cloud, Heroku, CI/CD)
    - Monitoring (Rollbar, Logrocket)
    - BI (Fullstory, Metabase)
    - Third party API & SDK integrations
- Founder, CEO at CodeMode (2018-09 - 2020-04)
    CodeMode is a tech consulting firm specializing in digital transformation for enterprises. We provide services such as user research, visual design, and software development to modernize and automate operations.
    
    - Generated 500k of revenue in the first 18 months of operation
    - Lead a team of 5 designers, developers, and product managers
    - Worked with established clients in a variety of industries such as SGS Canada, Consult IMI, and TechTO

EDUCATION:
- Unionville High School
- University of Toronto — Bachelor of Applied Science (B.A.Sc.), Engineering Science (2012-01 - 2016-01)

SKILLS:
(none)


========== P18 — position: dwelly ==========
HEADLINE: Software Engineer | Kotlin, Angular, GCP, Kubernetes
LOCATION: Grafenschachen, Burgenland, Austria

SUMMARY:
Hey there! I'm [name], a Software Engineer who loves to drive innovation through full-stack development and scalable architecture solutions.

Here's what I've done so far:
• 🔄 Microservice Migration: Modernized monolithic systems to scalable microservices, improving platform flexibility and future-proofing our architecture.
• 🌍 Geospatial Innovation: Enhanced core mapping functionalities to deliver intuitive visualizations of complex supply chain data for global clients.
• 🔌 API Ecosystem Growth: Built a developer portal and optimized APIs to streamline third-party integrations for enterprise customers.
• 🤝 Client-Centric Solutions: Spearheaded integrations with diverse external data providers, tailoring solutions to meet unique industry requirements.

Outside of work, I like to spend my time improving my skills by taking on new challenges, learning new languages and frameworks, contributing to open source projects, or just hanging out with friends and family. I am also quite interested in linguistics.

EXPERIENCE:
- Intermediate Software Engineer at Prewave (2024-04 - present)
    As an Intermediate Software Engineer at a leading Supply Chain B2B SaaS startup, I've spent my time driving innovation and enhancing our platform's capabilities. My work spans a wide range of critical projects, including:
    
    • Played a key role in the microservice migration initiative, contributing to the modernization of our architecture and improving scalability.
    
    • Contributed to core mapping functionalities, improving geospatial data visualization for our clients.
    
    • Enhanced our platform's API capabilities and established a developer portal, facilitating better integration for our API customers.
    
    • Worked on multiple high-profile client integrations for external data providers, demonstrating adaptability to diverse client needs.
- Junior Software Engineer at Prewave (2022-07 - 2024-04)
- Software Engineer at x.news information technology gmbh (2019-07 - 2019-07)
    During my internship at x.news, I gained valuable experience in both backend and frontend development, contributing to significant projects and technology transitions:
    
    • Utilized Java, Spring framework, and Hibernate to develop and maintain robust server-side applications.
    • Worked extensively with Knockout.js to build dynamic and responsive user interfaces.
    • Played a key role in the company's frontend technology transition, assisting in the migration from Knockout.js to Vue.js.

EDUCATION:
- HTL Pinkafeld — Matura, Informatik
- Heriot-Watt University — Master of Science - MSc, Computer Science

SKILLS:
Microservices, ArgoCD, Google Cloud Platform (GCP), NoSQL, Kubernetes, Nix, Java Virtual Machine (JVM), Database Systems, Angular, Supply-Chain-Software, Kotlin, Spring Boot, PostgreSQL, AngularJS, TypeScript, Spring Framework, Backend-Entwicklung, Programmieren, Node.js, Frontend-Entwicklung, Recherche, Microsoft Excel, Microsoft Office, JavaScript, Java, Microsoft Word, HTML, Cascading Style Sheets (CSS), Microsoft PowerPoint, C (Programmiersprache), C++, Problemlösung, Englisch, Switching, Kryptowährung, [link], Computersprache, Linux, Software, Rust, SAP-Produkte, Softwareentwicklung, Go


========== P19 — position: owner ==========
HEADLINE: Built Replit Agent — the first AI agent shipped to millions of users
LOCATION: United States

SUMMARY:
(none)

EXPERIENCE:
- Engineering Lead at Replit (2023-09 - present)
    Creator of Replit AI Agent — the first AI agent product shipped to millions of users
    Engineering Lead of AI team
    Reached $100M ARR (10x growth) in 6 months
- Machine Learning Engineer at ByteDance (2021-05 - 2022-05)
    Building machine learning algorithms on large-scale AI systems for recommendation and ads, to empower both TikTok and other Bytedance products.
- Senior Software Engineer at Google (2016-08 - 2021-04)
    User Modeling & Stateful Search. Building machine learning algorithms for Google's largest user models that empowers personalizations in Google Search, Discover and News.
- Research Intern at Microsoft (2016-03 - 2016-06)
    System Group, working on large-scale machine learning system.
- Co-Founder at Stealth Startup (2022-06 - 2023-09)
    Started a ventured-backed early stage AI startup doing workflow automation and AI code generation

EDUCATION:
- Beihang University — Bachelor of Engineering (BE), Software Engineering, Software Engineering (2012-01 - 2016-01)

SKILLS:
CS224N: Natural Language Processing with Deep Learning, CS231n: Deep Learning for Computer Vision, CS223A: Introduction to Robotics, CS221: Artificial Intelligence: Principles and Techniques, Algorithm Design, Mathematics, Software Development, Data Structure, Machine Learning, Computer Graphics, Data Analysis, C++, Python, C#, Java, C, MySQL, Software Engineering


========== P20 — position: owner ==========
HEADLINE: Software Engineer | Deepgram Labs
LOCATION: United States

SUMMARY:
Software Engineer with a bachelors degree in Computer Science from Madonna University…

EXPERIENCE:
- Software Engineer at Deepgram (2025-09 - present)
    Software Engineer working on the DG Labs team at Deepgram, a cutting edge voice AI Company. In DG Labs we strive to push the boundaries of what is possible with voice AI and AI in general.
-  at Deepgram ( - present)
- Junior QA Developer at iRule (2018-04 - 2018-12)
    As a junior QA developer, my responsibilities included creating and maintaining automated test scripts to test our application. I also had the opportunity to assist with feature implementations and bug fixes in the UI. Technologies used: Katalon Studio, Java, Groovy, Selenium, Angular 4
- Software Engineer at Ford Motor Company (2019-01 - 2025-08)
    Contract Software Engineer at Ford Motor Company via Brooksource working in  Global Data and Analytics. I work on an application to assist marketing when building campaigns and identifying potential customers. In this project, I have had the opportunity to lead the UI development, and work on all aspects of the applications.  Technologies used: Angular 7, Spring Boot, MySQL, Hadoop, Pivotal Cloud Foundry, and Jenkins.
- IT Intern at Maxion Wheels (2015-08 - 2018-04)
    Supported an office of eighty plus users, performing both hardware and software repairs. Supported global help desk, and assisted front end team in development of Microsoft System Center Service Manager. Worked with advanced engineering as lead software developer to develop smart sensor applications for wheels and plant floors. 
- Technical Sales Associate at Canton Computers (2014-04 - 2015-07)
    Sales Associate and Technician at the Livonia branch of Canton Computers. Was responsible for checking in systems for service, on the spot diagnostic work, as well as hardware and software repair.

EDUCATION:
- Madonna University — Bachelor's degree, Computer Science (2014-01 - 2018-01)

SKILLS:
Programming, Soccer Coaching, Music Theory, LabVIEW, Python, C++, Microsoft SQL Server, Git, Presentations, MySQL, Pivotal Cloud Foundry (PCF), Java, Angular 7, Hadoop


========== P21 — position: owner ==========
HEADLINE: Helping Enterprises Transform Customer Experience with Voice AI | Strategic AE at PolyAI | Conversational AI | CX Automation | AI-Powered Call Center Solutions
LOCATION: Mesa, United States

SUMMARY:
I’m an enterprise sales leader with over 25 years of experience helping organizations navigate digital transformation in customer service. At PolyAI, I partner with large, complex businesses to deploy conversational voice AI that drives measurable results from reducing call volume to improving customer satisfaction.

My approach is grounded in partnership and precision. I work closely with stakeholders across CX, IT, and operations to identify the right use cases, align on KPIs, and build a roadmap for success.  If you're exploring voice automation, AI-powered support, or ways to improve efficiency at scale, I'd be happy to connect.

EXPERIENCE:
- Strategic Account Executive at PolyAI (2024-03 - present)
    Dedicated to assisting companies in reducing operational costs while enhancing customer and employee experience through the utilization of PolyAI's Next Generation AI Virtual Assistant.
- VP, Enterprise Sales at Drips (2020-03 - 2024-02)
    At [link] one of the fastest-growing marketing software platforms, we run automated voice, sms, and email drip marketing campaigns to convert more of your leads to calls.  You can get more qualified calls and less complaints with our compliant, tasteful, and effective drip campaigns.
- SR Sales Executive at Critical Path Software (1998-07 - 2000-11)
- Enterprise Sales Manager at NICE Ltd (2005-06 - 2011-06)
    NICE systems is the global provider of advanced solutions that help organizations extract meaningful insight from interactions. Customer calls and other multimedia interactions (such as emails) contain information that, when analyzed, offers the organization valuable insights, which help improve the decision-making process and drive overall business performance.
- Sales Director at Enkata (2011-07 - 2012-01)
    Enkata is a private company providing on-demand applications that improve the performance of people-intensive organizations such as call centers, claims processing operations and sales organizations by relating each individual’s performance to top level goals and metrics.
- Vice President at Mattersight Corporation (2012-01 - 2013-02)
    Mattersight is a leader in enterprise analytics focused on customer and employee interactions and behaviors. Mattersight's Behavioral Analytics service captures and analyzes customer and employee interactions, employee desktop data, and other contextual information to improve operational performance and predict future customer and employee outcomes. Mattersight’s analytics are based on millions of proprietary algorithms and the application of unique behavioral models. The company's SaaS+ delivery model combines analytics in the cloud with deep customer partnerships to drive significant business value. Mattersight's applications are used by leading companies in Healthcare, Insurance, Financial Services, Telecommunications, Cable, Utilities and Government. See What Matters™ by visiting [link]
- Sales Director at Enkata (2011-07 - 2012-01)
    Enkata is a private company providing on-demand applications that improve the performance of people-intensive organizations such as call centers, claims processing operations and sales organizations by relating each individual’s performance to top level goals and metrics.
- Sales Director, Western Region at Uniphore (2020-01 - 2020-03)
    Uniphore is a global Conversational AI technology company that delivers transformational conversational AI solutions for businesses. Uniphore offers solutions for Conversational Analytics, Conversational Assistant, and Conversational Security. Our solutions improved Customer Experience through AI and delivers dramatic efficiency gains and expense reduction at Contact Centers.
    
    Uniphore has offices worldwide including the USA, Asia Pacific, and India.
- Vice President at Mattersight Corporation (2012-01 - 2013-02)
    Mattersight is a leader in enterprise analytics focused on customer and employee interactions and behaviors. Mattersight's Behavioral Analytics service captures and analyzes customer and employee interactions, employee desktop data, and other contextual information to improve operational performance and predict future customer and employee outcomes. Mattersight’s analytics are based on millions of proprietary algorithms and the application of unique behavioral models. The company's SaaS+ delivery model combines analytics in the cloud with deep customer partnerships to drive significant business value. Mattersight's applications are used by leading companies in Healthcare, Insurance, Financial Services, Telecommunications, Cable, Utilities and Government. See What Matters™ by visiting [link]
- Enterprise Account Executive at PolyAI (2024-02 - 2024-03)
    PolyAI builds customer-led conversational assistants that carry on natural conversations with customers to solve their problems. Our voice assistants understand customers, regardless of what they say or how they say it.
    
    We serve enterprises where brand experience and accurate resolutions are essential to doing business. Our customers include some leading names in banking, hospitality, insurance, retail, and telecommunications. 
    
    Our enterprise clients deploy PolyAI voice assistants to cut wait times and free up live staff to focus on calls requiring empathy and judgment. As a result, our clients see improved customer satisfaction, employee retention, and operational efficiency
- Senior Sales Director, Retail & Technology at Interactions LLC (2013-02 - 2020-01)
    Interactions enables businesses to more effectively interact with their customers. The Company’s patented technology integrates an unprecedented level of understanding into automated voice, mobile, and Web systems -- enabling a productive two-way dialogue, quick and efficient responses to customer requests, and a natural and easy way to communicate.
    
    Interactions builds and deploys automated Virtual Assistant applications that utilize industry-leading natural language technology, engaging callers in a manner that directly mirrors a live conversation.
- Sr Sales Executive at Centergistic Solutions (2000-11 - 2005-06)
    Centergistic Solutions sells Real-Time and Historical Analytic solutions for the contact center industry. They currently serve customers like ABN AMRO, American Express, WellPoint BCBS and over 4000 other companies worldwide.

EDUCATION:
(none)

SKILLS:
Analytics, Software Sales, Call Center, Solution Selling, SaaS, Enterprise Software, Predictive Analytics, Customer Analytics, Business Analytics, Sales Analytics, Text Analytics, Customer Experience, Customer Satisfaction, Customer Retention, Net Promoter Score, Back Office, Customer Analysis, Software Industry, Call Centers, Business Intelligence


========== P22 — position: owner ==========
HEADLINE: Software Engineering Leader ∣ Scaling High-Growth Technology | Building Infrastructure for the Agentic Economy
LOCATION: United States of America

SUMMARY:
I am a High-Impact Engineering Leader with extensive experience building, modernizing, and scaling high-velocity development organizations. I specialize in driving the successful evolution of monolithic applications into scalable, cost-efficient, cloud-native architectures—primarily on Azure, GCP, and AWS. My technical expertise centers on foundational system design, implementing robust CI/CD pipelines (Kubernetes, Terraform, Docker), and leveraging event-driven microservices to accelerate business value and improve operational stability.

My core value is my ability to manage complexity at scale, both technically and organizationally. I have a proven track record of building and retaining high-performing, distributed engineering teams. I excel at establishing clear technical roadmaps, optimizing organizational efficiency through Agile standardization, and mentoring future leaders to ensure sustained team growth and high output.

I am actively seeking Engineering Manager and Director roles where I can apply a strategic, results-driven approach to complex integration and cloud migration challenges. I am passionate about leading technical teams to translate architectural excellence into measurable business impact, such as reducing operational overhead and enabling faster time-to-market for critical product lines.

EXPERIENCE:
- Founding Engineer - Core Platform at PayOS (2025-11 - present)
- Senior Director of Engineering at IXOPAY (2022-10 - 2025-10)
- Engineering Manager / Engineering Lead at Kount (2020-01 - 2021-08)
    My team is responsible for the technical architecture, implementation and support for the software built to integrate into eCommerce platforms for Shopify, Big Commerce, Woo Commerce, Magento and Sales Force Commerce Cloud.
    
    Collaborate with the Sales, Marketing and Product teams to Identify target platforms where an extension can help accelerate revenue growth for the company
    
    Lead software engineering teams leveraging Kount engineers and contractor resources as appropriate to build integrations into the target platforms
    
    Key Achievements:
    Led the effort to build an app for the Big Commerce marketplace.  
    
    Leading the effort to build an extension for the Woo Commerce marketplace utilizing the evolving technology stack that facilitated a successful launch of the Big Commerce App.
- Senior Solutions Engineer at Kount (2016-12 - 2020-01)
    Supported Kount Sales Executives with their technical interactions with prospects throughout the sales process
    
    Key Achievements:
    Led the successful project to build an app for the Shopify platform.  The app has been well adopted by a large number of Shopify shops and continues to feed the sales pipeline with prospects that would have otherwise been unable to use Kount.
    
    Collaborated with the Customer Success team to build out an onboarding plan for new customers and implemented a Jira workflow to improve the support of customers throughout the life of their contract.
    
    Built an application to allow the Solutions Engineering teams to quickly create custom demos for prospects with unique use cases
- Early Career: Foundational Engineering & Consulting at Various (1993-10 - 2008-10)
    Strong Backend Foundation built through complex systems development and technical consulting, driving mission-critical software delivery for high-profile private and public sector clients (Microsoft, Hewlett Packard, DHS).
    
    Highlights include:
    
    • Technical Leadership & Mentorship: Established a track record of building and developing talent, providing technical guidance to junior engineers and senior peers alike, including managing a remote team of two and leading staff on software best practices.
    
    • Strategic Program Funding: Built mission-critical software prototypes and led technical demonstrations for the Department of Homeland Security, directly enabling the University to secure a 5-year research grant.
    
    • Complex Problem-Solving: Leveraged diverse technologies (C#/.NET, Java, C, PowerBuilder) to solve complex problems and improve business efficiencies across private and public sector engagements.
- Software Engineering Manager at Bodybuilding.com (2011-10 - 2016-12)
    •	Lead Software Engineering teams, architecting and building complex software solutions in C#/.NET using SQL Server relational databases.  The distributed software applications utilize NServiceBus with RabbitMQ as the transport layer.
    •	Work closely with Business Stakeholders, Engineering Leadership, Project Management and Product Managers to align 2 week Sprints and identify key deliverables required to execute against the company roadmap
    •	Utilize the Agile software development methodology and its key principles, like Daily Scrum, Sprint Planning, Backlog Grooming, Release Planning and Retrospectives to optimize the efficiencies of software engineering teams
    •	Responsible for the career development of 7 software engineers, 2 quality assurance engineers and 1 SDET 
    •	The position reports to the Vice President of Engineering, Commerce 
    
    Key Achievements:
    •	Integrated with a 3rd Party Logistics (3PL) provider in the Netherlands.  The project went from requirements gathering, project plan, software architecture design, implementation and deployment in a little over 5 months 
    •	Led a team of software engineers responsible for building a custom Warehouse Management System (WMS) software application.  The project included a software implementation that took about 5 months, and a phased rollout to 5 fulfillment centers
    •	Led a team of software engineers responsible for preparing and migrating the custom WMS software application to Amazon Web Services (AWS) as a key milestone in a project to launch a fulfillment center in the United Kingdom    
    •	Oversaw the software engineering team implementation of a redesign of custom Order Management System (OMS) software.  The goal of the project was to build the application so it could be more data driven allowing the Logistics team to more efficiently manage carrier rate shopping and fulfillment center allocation rules using strictly data uploads
    •	Integrated with Logility, a 3rd party Supply Chain Management application
- Senior Software Engineer at Bodybuilding.com (2008-10 - 2011-10)
    •	Proposed architecture utilizing software engineering best practices with an eye on performance, quality and testability
    •	Worked on a team with as many as 4 software engineers, and 2 QA engineers writing software to solve complex problems improving efficiencies for business stakeholders, and improving customer experience
    •	Collaborated with system engineers and software engineering teams to define companywide engineering standards, and built software that adhered to those standards
    •	Reported to the Director of Software Engineering, Commerce
    
    Key Achievements:
    •	Built software to integrate with Microsoft Dynamics Great Plains to allow the finance team to be able to accomplish their job more effectively 
    •	Designed and built custom Order Management System (OMS) software in C#/.NET in a little more than 4 months.  The software integrated with a new ecommerce platform that was implemented in parallel and legacy Warehouse Management System (WMS) software
- Engineering Manager at Truepill (2021-08 - 2022-10)

EDUCATION:
- Rochester Institute of Technology — Bachelor of Science (BS), Computer Engineering (1988-01 - 1993-01)

SKILLS:
C#, .NET, Agile Methodologies, XML, Scrum, Web Services, Software Development, Java, SQL, Software Engineering, HTML


========== P23 — position: dwelly ==========
HEADLINE: Product Engineer at Granola
LOCATION: London Area, United Kingdom

SUMMARY:
Software Engineer working on Avatars & Identity at Meta.

I have a passion for product development, algorithms, solving complex problems and improving performance of systems. I’ve worked in various areas like static analysis tooling, account security user products, ads and conversational products and most recently I have been working on avatars

EXPERIENCE:
- Product Engineer at Granola (2026-07 - present)
    Working on agentic workflows at Granola
- Staff Software Engineer at Meta (2023-08 - 2026-07)
    Avatar Editors
- Senior Software Engineer at Meta (2022-03 - 2023-08)
    Building creation & editing experiences in the Avatars team
- Software Engineer at Meta (2021-11 - 2022-06)
- Software Engineer at Facebook (2018-08 - 2021-11)
- Practical demonstrator at University of Oxford (2018-01 - 2018-04)
    Work as a demonstrator in practical sessions in concurrent programming ( 2nd year course).
- Software Engineering Intern at Facebook (2017-06 - 2017-09)
    Part of the Account Security Products team. 
      - Hack(PHP)
      - React
      - Data analysis
- Software Engineering Intern at Semmle (2016-06 - 2016-10)
    Worked on a project aimed at adding TypeScript to the language support of the company together with standalone projects involving query writing in the company’s language(QL). Most of the work involved understanding of the TypeScript compiler and compiler API and JavaScript’s syntax in order to do the static analysis
- Raw Data Analyst at  (2015-07 - 2016-05)
    Analyzing raw data that is being output by the product of the company before being shown to customers, reporting any discrepancies and suggesting reasons and solutions for bugs that may appear.

EDUCATION:
- University of Oxford — Masters, Computer Science

SKILLS:
Research, Programming, Works well in a team, C, Computer Science, Matlab, Science, C++, JavaScript, Algorithms, Hack, Computer Graphics, Teamwork, C (Programming Language), PHP, React.js, Software Development, Data Structures


========== P24 — position: owner ==========
HEADLINE: Founding Software Engineer
LOCATION: San Francisco, United States of America

SUMMARY:
(none)

EXPERIENCE:
- Founding Software Engineer at LangChain (2023-05 - present)
    Helping developers build with LLMs
- Software Engineer at Google (2015-08 - 2016-10)
    Created a better sharing experience for photos.google.com
- Software Development Intern, ICU Team at Epic (2014-05 - 2014-08)
    Built features for an iPad app (Epic Canto) helping doctors quickly view patient information
- Co-Founder and CTO at Autocode (2022-01 - 2023-01)
- Co-Founder at Autocode (2016-11 - 2022-01)
    An online code editor with API autocomplete, instant hosting, and a Standard Library anybody can contribute to. Sync data, build bots and customize workflows.
- Consultant at Remora Software, LLC (2023-02 - 2023-05)
    Helped companies design, build, and scale great software

EDUCATION:
- Princeton University — A.B., Computer Science

SKILLS:
iOS development, Computer Science, Objective-C, Java, C, Python, Django, Photoshop, Tutoring, Web Design, Squash, JavaScript, Google Closure, Leadership


========== P25 — position: owner ==========
HEADLINE: Senior Staff Platform Engineer | TypeScript, Cloud Computing
LOCATION: Basalt, United States

SUMMARY:
Senior Staff Platform Engineer at Thoughtful AI with extensive experience in designing and delivering scalable software systems and cloud-native platforms. With a strong foundation in TypeScript, cloud computing, and Kubernetes, contributed to the development of a robust platform that enables seamless deployment and management of applications. Notable contributions include implementing a Kubernetes-based platform and enhancing open-source Knative Eventing capabilities for AWS integration.  Focused on creating developer-friendly architectures and fostering cross-functional collaboration to solve complex technical challenges. Dedicated to establishing engineering standards, mentoring teams, and supporting long-term business growth through impactful technical leadership and innovative platform solutions. Passionate about building scalable and reliable systems that empower engineers and support evolving organizational goals.

EXPERIENCE:
- Software Developer Engineer at Pluck (2005-11 - 2007-07)
    • Wrote product from small collection of ideas to a working demo in 6 weeks using C# and ASP.Net. Turned that into a beta product in another 2 months.
    • Implemented Javascript based widget delivery system for customer web sites. This system served over 5 million widgets a day.
    • Transformed site from ASP.Net to Monorail with NVelocity templates. This separated the business logic from the UI, eased the use of AJAX on pages, and allowed less technical web developers to work on the UI.
    • Optimized memory usage using WinDbg
    • Designed and implemented ETL workflow using SQL Server 2005 SSIS. This workflow provided data for traffic reporting and customer billing.
- Senior Software Developer at Advanced Solutions International (2003-11 - 2005-09)
    • Tech Lead on integral feature for application platform. Worked with business analysts and product manager to ensure correct features were delivered as scheduled. Oversaw technical issues, tasked team members, and validated deliverable quality through code reviews, unit testing, and feature testing.
    • Design team member. Group of six engineers and one manager tasked with addressing development process and architecture issues raised by team members and other engineering staff.
    • Automated the server-side generation of Word documents by using WordML (the XML description of a Word document) and transformed it into XSL to enable the merging of data into the document.
    • Implemented feature using Domain Specific Language (DSL) tools built into the application platform for data access, queries, workflow, and UI. Business logic and unit tests were developed using C#.
    • Wrote Visual Studio .Net add-in to generate standardized code based on information specified within product. This greatly increased productivity for all developers and reduced bugs.
    • Speaker at annual reseller conference. Lead sessions introducing how to extend our product using the application platform included with the product.
- CTO at Daily Number (2017-04 - 2018-11)
    • Architected and implemented daily fantasy sports platform. Balanced need to deliver to market quickly on a small budget against long term technical debt and scalability. Technology stack includes Node.js, Python, PostgeSQL, MongoDB. 
    • Managed remote contractors ensuring quality work was delivered that met the contract scope.
    • Maintained and upgraded legacy Meteor code base (acquired from a previous company as a proof of concept before my joining the company) for Cordova mobile app.
- Software Developer at Austin Info Systems (2001-10 - 2003-11)
    • Designed and developed business and data access layers for n-tier application using C#. Implemented unit tests for all features using NUnit. Worked with GUI team members to develop interface, and other application developers to incorporate interoperability between the .NET application and legacy COM applications.
    • Architecture team member. Responsibilities include identifying deficiencies in current products, reviewing application requirements and designs to ensure core goals of the system were being met, and addressing developer concerns as they arose.
- Sr. Architect at TravelBoss (2018-12 - 2020-04)
    • Proposed and implemented sustainable and testable architecture patterns for React/Redux application using Hooks, Redux Saga, Storybook, Jest, and Enzyme.
    • Developed Node serverless functions with automated deployments to AWS Lambda and API Gateway using Serverless and Terraform.
    • Automated AWS deployment of staging and production React application using Terraform and CircleCI. By using S3 and CloudFront, users were not impacted by production deployments of cache-busting, code-split javascript modules.
    • Used behavior driven development practices to establish company wide expectation of documenting feature requirements through unit testing for both server and front end code.
    • Mentored junior engineers on best patterns and practices for software development.
- Software Architect at Double Line, Inc. (2011-06 - 2015-07)
    Software Architect (September 2014 - July 2015)
    Senior Software Developer (January 2013 - September 2014)
    Contractor (June 2011 - Jan 2013)
    
    • June 2014 - July 2015: Architected and led implementation of system to generate the Ed-Fi data standard artifacts from a single source. The metadata is stored in a custom domain specific language that was implemented with ANTLR. The DSL then builds the Ed-Fi XSD, ODS DDL and other metadata required for the REST API allowing for a single source of truth across different data representations. Managed team of ten developers through kanban process. Worked directly with customer to determine the product’s feature priorities and scope of work. 
    • Jan 2014 - May 2014: Reworked the Ed-Fi XSD data standard and ODS schema to support the automatic code generation needs of the Ed-Fi REST API. The culmination of this effort is the Ed-Fi 2.0 standard, released April 2015. This work required quickly learning in depth needs of the various facets of the domain and negotiating with stakeholders on how to implement needed changes. Established naming conventions and design patterns to be used in all future development and extension of the data standard.
    • Nov 2012 - Jan 2014: Developed application to allow developers, business analysts, and clients configure metadata used to power web dashboard. Technology stack includes Bootstrap 3, Knockout, Knockback, Backbone, Backbone Relational, jQuery, REST web application, ASP.Net MVC, Automapper, code-first Entity Framework 5, and StructureMap.
    • June 2011 - Nov 2012: Worked on the Ed-Fi Dashboard, an application providing up to date educational metrics to teachers, staff and administrators in five pilot districts in Texas with support from the Dell Foundation. This application is now deployed in five states and is freely available through a license from the Dell Foundation. Technology stack includes jQuery Templating, jQueryUI, jQuery, REST web application, ASP.Net MVC, SubSonic, and Castle Windsor.
- Software Engineer at Pervasive Software (2001-01 - 2001-06)
- Principal Software Developer at TuffWerx, LLC (2007-08 - 2009-09)
    • Architected and coded web application using C# and the .Net 2.0 Framework. The front end utilized Monorail and the database access was done with NHibernate on a MySQL database.
    • Developed build system using NAnt to help developers with configuration and to deploy to both staging and production servers. NUnit unit tests are run as a part of this process.
    •  Integrated with third-party applications such as Amazon's S3 REST interface, Ebay's  SOAP API, and SalesForce's SOAP API.
    •  Launched a production site from a business idea in seven months on limited budget with one other part-time developer.
    •  Developed SalesForce application to allow tracking of TuffWerx specific data inside SalesForce using Apex, Visual Force, and the SalesForce web service API.
- Engineering Manager at Ahana (2020-05 - 2024-06)
    • Managed a remote team of software engineers. Responsible for hiring, resource allocation, and  employee performance.
    • Architected control plane of Ahana Cloud using a system of microservices utilizing cloud native offerings of AWS such as Lambdas and Elastic Container Service. Lead engineer for implementation of this Node.js, Reach, and Python system. Responsible for taking ideas from Product Management and generating technical and UX requirements. Guides and mentors team members working on implementation.
    • Developed Ahana Cloud compute plane. The compute plane is programmatically deployed in a Kubernetes cluster residing in the customer’s VPC.
    • Worked with sales, product management, and customer success in presales engagements and contract renewal discussions. Solutioned sustainable features quickly into production in order to close important deals.
    • Provided company leadership regular updates of initiative progress. Managed shifting priority set and limited resources to deliver features when needed while minimizing interruptions to engineers workstreams.
    • Responsible for engineering and IT department reports during due diligence for IBM acquisition. Participated in meetings between Ahana and IBM to answer clarifying due diligence questions.
    • After IBM acquisition, worked to transition Ahana business to IBM. Assisted customers with migration efforts and worked to ensure exceptional product experience continued.
- Senior Software Developer at Advanced Solutions International (2010-08 - 2011-05)
    • Identified application performance hotspots and fixed problems found.
    • Unified code duplicated in to product branches into a single branch and incorporated the branch into the automated build process for both products.
- Senior Staff Platform Engineer at Thoughtful AI (2024-07 - 2026-04)
    • Architected and implemented Kubernetes based platform enabling Thoughtful engineers to easily deploy and run applications. Built on EKS using Karpenter to manage compute, Istio Ambient for mTLS support, APISIX for ingress and api gateway, KNative Serving for managing applications and functions, KNative Eventing for event based message delivery.
    • Added support for AWS MSK to OSS Knative Eventing Kafka plugin. Enhanced OSS Knative Eventing to support k8 pod default IAM credentials for AWS integration sources and sinks. Both PRs plus supporting documentation PRs merged to the OSS project.
    • Established a data lakehouse using AWS Glue, Apache Iceberg, PySpark, S3, Athena. Ingested multiple data sources including Aurora Postgres through AWS DMS, Jira, Zoho, and Bugsnag. The lakehouse greatly reduced the load on our database by moving QuickSight query loads to Athena.
    • Mentored team members, reviewed architecture proposals
    • Designed and implemented central authentication and authorization service for all Thoughtful applications. Written in Node using Stytch for the IdP plus custom APISIX plugins and CloudFront lambdas.
- Architect / Lead API Developer at Hop Market (2015-07 - 2017-04)
    • Architected and implemented social commerce system using a behavior driven design style to allow for self documenting unit and integration tests. Technology stack includes C#, Entity Framework, StructureMap, AutoMapper, OData. 
    • Designed the API for mobile and web clients while operating in a continuous deployment environment. This adds complexity because existing published functionality must be maintained while development for new features is progressing. 
    • Implemented administrative web site giving non-technical staff access to application data in an easy to consume manner. Technology stack include React, Redux, OData, Webpack, ES6. 
    • Created a chatbot using MS Bot Builder, LUIS and our existing API. The bot was an interesting exploration into what a conversational interface could provide for new and existing customers.
- Contractor at MessageOne (2008-09 - 2009-04)
    • Refactored existing business logic into a new API to allow multi-threaded use while maintaining backwards compatibility. Designed and developed new WinForm app to use the new API to allow multi-threaded processing with stop and resume functionality.
    • Performed maintenance work and responded to customer issues on existing products.
    • Added calendar import support for existing Lotus Notes extension and improved calendar import functionality for Outlook.

EDUCATION:
- Trinity University — BS, Computer Science, Math (1997-01 - 2000-01)

SKILLS:
TypeScript, Cloud Computing, Software Development, Software Architecture, Full-Stack Development, Infrastructure as a Service (IaaS), Domain-Driven Design (DDD), JavaScript, Node.js, Kubernetes, REST APIs, Amazon Web Services (AWS), Presto, AWS Glue, Apache Spark, React.js, Amazon EKS, Knative, Terraform, AJAX, C#, SQL, Agile Methodologies, ASP.NET, .NET, LINQ, Visual Studio, Microsoft SQL Server, Web Development, HTML, SOA, Java, Web Applications, XML, T-SQL, Scrum, Unit Testing, NUnit, Test Driven Development, Web Services, Requirements Analysis, Software Project Management, Software Engineering, REST, Software Design, Multithreading, Subversion, SDLC, MySQL, Databases, SSIS, Database Design, Enterprise Architecture, SOAP, Architecture, User Interface, NHibernate, CSS, Agile Project Management, Object Oriented Design, Business Analysis, .NET Framework, Service-Oriented Architecture (SOA)


========== P26 — position: owner ==========
HEADLINE: Engineer at Replit
LOCATION: San Francisco, United States

SUMMARY:
Teaching AI taste

EXPERIENCE:
- Member of Technical Staff at Replit (2024-01 - present)
- Software Engineer at Zywave (2020-07 - 2022-02)
    Clariondoor - acquired by Zywave
    • Continually built new features for flagship Quoting Portal web platform built on Vue.js, PHP Laravel, and Amazon Web Services, used by insurance professionals around the globe to achieve millions of quotes and counting
    • Worked closely with project managers, business analysts, and massive insurance clients to gather requirements and develop custom frontend tools and backend APIs to integrate with existing customer platforms
- Campus Tour Guide at University of California, Santa Barbara (2018-01 - 2019-11)
- Front Desk Attendant at UCSB - Housing and Residential Services (2018-09 - 2018-11)
- Software Quality Assurance Engineer at Graduate Division at UCSB (2017-06 - 2017-12)
- Software Engineer at Praevium Research, Inc. (2019-03 - 2020-03)
    Full-Stack Developer
    • Built web applications from the ground up for company that designs and fabricates semiconductor lasers
    • Utilized MVC framework (Javascript, PHP, HTML/CSS, Bootstrap, various APIs, & SQL) to design databases and UX/UI for graphical visualization of hardware data
    • Developed apps in Python to communicate with field instruments and industrial devices over a network
- Fullstack Software Engineer at Lendtable (2022-02 - 2024-01)
    Product, Design, and Payments at Fintech Startup

EDUCATION:
- UC Santa Barbara — Bachelor’s Degree, Computer Engineering

SKILLS:
Product Engineering, Full-Stack Development, Product Design, Design Engineering, Front-End Development, TypeScript, JavaScript, Next.js, React.js, Node.js, Tailwind, Amazon Web Services (AWS), Back-End Web Development, Python, HTML/CSS, Computer Engineering, Public Speaking, Java, Google Cloud Platform (GCP)


========== P27 — position: dwelly ==========
HEADLINE: Engineering at Legora
LOCATION: Copenhagen, Capital Region of Denmark, Denmark

SUMMARY:
(none)

EXPERIENCE:
- Member of Technical Staff at Legora (2025-11 - present)
    Documents team
- Senior Platform Engineer at Pleo (2024-11 - 2025-10)
    - Led parts of the migration of our whole observability stack from Datadog to Grafana Cloud using OpenTelemetry
    - Designed and implemented robust observability data pipelines, enabling traces sampling and metrics generation from traces
    - Built an internal workflow engine for our platform team to run critical one-off tasks - used to orchestrate the migration of 50+ databases in production
    - Designed our new cross-cloud IAM permissions model
    - Part of our 24/7 on-call rotation for all our services
- AI Engineer at Pleo (2023-06 - 2024-11)
    - Built multiple RAG agents with Python to enhance the productivity of both the sales and customer support functions
    - Deployed LLMs and embedding models in our Kubernetes clusters for non-critical production workloads using HuggingFace’s open source inference servers
    - Built a micro-service for expense auto-categorisation leveraging an open-source embedding model and a vector database
    - Worked on a POC to generate spending guidelines using Generative AI
- Site Reliability Engineer at Pleo (2022-06 - 2023-06)
    - Led initiatives to redesign our entire CD pipelines to a GitOps approach using FluxCD
    - Planned and executed the migration of all our Kubernetes clusters to our fully codified GitOps-based infrastructure
- Junior Back End Engineer at Pleo (2021-10 - 2022-06)
    - Worked on two different micro-services responsible for the billing infrastructure and management of entitlements (which feature is accessible for which market and on which plan, etc.)
    - Learned how to work in a micro-services environment using asynchronous messaging
- Machine Learning Engineer at Botpress (2019-05 - 2019-09)
- Intern Machine Learning Engineer at Botpress (2019-05 - 2019-09)
    - Worked on creating a new natural language understanding pipeline using multiple SVMs, a k-means and CRFs
- Product Engineer at Snipcart (2016-05 - 2018-08)
    - Worked on the back-end monolith to integrate with payment, shipping  & tax providers, and the handling of cart sessions. I also worked on the FE for both the store dashboard and the injected cart
    - Worked on a wide variety of other tasks: writing technical blog posts, doing customer support, SEO & product scoping/discovery

EDUCATION:
- Université de Montréal — Bachelor's degree, Mathematics and Computer Science

SKILLS:
Pandas (Software), Google Cloud Platform (GCP), Go (Programming Language), Software Observability, OpenTelemetry, Hugging Face Products, Large Language Models (LLM), Python (Programming Language), Retrieval-Augmented Generation (RAG), Terraform, CI, Continuous Delivery (CD), Datadog, Amazon Web Services (AWS), Kotlin, Microservices, Apache Kafka, Amazon SQS, Kubernetes, React.js


========== P28 — position: dwelly ==========
HEADLINE: Software Engineer at Synthesia
LOCATION: Acireale, Italy

SUMMARY:
I'm a software developer and I'm passionate about development and computer science in…

EXPERIENCE:
- Software Engineer at Synthesia (2022-12 - present)
    Synthesia develops AI-driven synthesis technology to empower companies to create realistic videos as easy as writing an email.
    
    Funded by Mark Cuban, LDV Capital, Seedcamp, MMC Ventures, Taavet Hinrikus, VAS Ventures, Nigel Morris and TinyVC. 💸 
    
    Used in campaigns with Lionel Messi, David Beckham, Snoop Dogg, Reuters, BBC, Accenture and more. 🔥
- Frontend Engineer at Helixa (2021-08 - 2022-12)
    I worked with React on a micro-frontends platform.
- Frontend Engineer at MOVIA SpA (2018-06 - 2021-07)
    I worked with React and Vue as a frontend engineer.
- Full Stack Engineer at M2D Technologies (2017-03 - 2018-06)
    I worked as a full stack engineer using Python (with Django) and Javascript.
- Full Stack Engineer at Braintech (2014-02 - 2017-02)
    Pane&Design is a a digital company based in Italy. We help startups and companies building successful experiences. Our headquarters are in Catania and Milan.
    
    My activities:
    
    - Web development (backend and user interaction)
    - Bug fixing
    - Wordpress development
    
    I have worked in several projects like e-commerces, auction sites and back offices.

EDUCATION:
- IISS "Galileo Ferraris" - Acireale — High school degree, Computer Science (2008-01 - 2013-01)
- Università di Catania — Bachelor's degree, Computer Science (2013-01 - 2017-01)

SKILLS:
JavaScript, React.js, React Native, Programming, Front-End Development, Web Development, Redux.js, Node.js, Git, HTML 5, CSS3, Python, jQuery, TypeScript


========== P29 — position: dwelly ==========
HEADLINE: Engineering at ElevenLabs | IITD CS
LOCATION: London, England, United Kingdom

SUMMARY:
I am interested in the intersection of Artificial Intelligence and Software Engineering, where one complements the other by compensating for the shortcomings. This can be in the form of AI-for-the-edge(light-weight computer vision, audio-based predictions, etc), machine learning for compiler optimization, high-performance architectures for distributed training and federated learning, among others.

I find high-performance backend development to be an interesting area under software engineering. For AI, my primary interests comprise of computer vision, speech processing and natural language processing, with explainable AI occupying a position of high importance overall.


EXPERIENCE:
- Software Engineer at ElevenLabs (2025-08 - present)
- Senior Software Engineer at LG Ad Solutions (2025-04 - 2025-07)
- Software Engineer 2 at LG Ad Solutions (2024-02 - 2025-03)
- Data Scientist at LG Ad Solutions (2021-07 - 2024-01)
- Artificial Intelligence Engineer at Bang & Olufsen (2020-01 - 2020-08)
    Designed an optimized convolutional neural network(CNN)  architecture to perform Acoustic scene classification on low-powered, low-memory devices such as headphones, to enhance and adapt the noise cancellation filters based  on listener's surroundings 
    Adapted the CNN model to perform high-accuracy inference on company-specific-microphone recordings, via transfer learning
    Developed end-to-end framework for rapid experimentation with CNN architectures, data preprocessing methods, transfer learning
    Reduced memory requirements further by pruning and quantizing the 32-bit float model to low-memory 8-bit integers and optimizing the input preprocessing and space requirements
- Software Engineer at American Express (2019-05 - 2019-07)
    Worked with the engineering team on the development of Enterprise Communications Platform (Raven)
    Collaborated to come up with an architecture for  and develop one of the components of Raven, following Agile methodology
    Built custom JSON and POJO parsers, which were generic and required no code changes when any change in requirements/specifications occur in future
- Machine Learning Researcher at Safe Security (2018-05 - 2018-07)
    Adversarial Machine Learning: Working on finding and exploiting vulnerabilities in various Machine Learning Models such as Support Vector Machines, Naive Bayes and Neural networks. Prepared a basic program to exploit vulnerabilities leading to a reduction in vulnerabilities of a Machine Learning Model.
    This model can then also be used to see how vulnerable a given neural network is to certain types of attacks.

EDUCATION:
- Indian Institute of Technology, Delhi — Master's degree, Computer Science and Engineering
- Indian Institute of Technology, Delhi — Bachelor's degree, Computer Science and Engineering

SKILLS:
Software Development, Data Structures, Java, Python, Machine Learning, Linux, Management, Data Analysis, Data Science, Programming, Teamwork, Time Management, Research, Research and Development (R&D), TensorFlow, Scikit-Learn, Tableau, Computer Science, C++, Deep Learning


========== P30 — position: dwelly ==========
HEADLINE: AI Engineer at Granola | Prev. Apple
LOCATION: London, England, United Kingdom

SUMMARY:
(none)

EXPERIENCE:
- AI Engineer at Granola (2025-05 - present)
- AI Engineer at Nothing (2024-12 - 2025-04)
- Machine Learning Engineer at Apple (2022-09 - 2024-12)
- Machine Learning Engineer at Apple (2021-09 - 2021-11)
- Software Engineer at Windmill (2021-06 - 2021-09)
- Machine Learning Engineer at Eaton (2021-01 - 2021-07)
- Intern at Huawei (2020-11 - 2020-11)
- Software Engineer at Houghton Mifflin Harcourt (2020-06 - 2020-09)
- Data Analyst at Stats Perform (2018-06 - 2018-09)

EDUCATION:
- Trinity College Dublin — Bachelor of Arts - BA, Computer Science
- Trinity College Dublin — Bachelor of Arts - BA, Computer Science
- Trinity College Dublin — Master of Computer Science, Machine Learning
- Trinity College Dublin — Master of Computer Science, Machine Learning

SKILLS:
Machine Learning Algorithms, Large Language Models (LLM), Generative AI, Machine Learning, Artificial Intelligence (AI), Data Science, Computational Mathematics, Probability, Statistics, Wireframing, Model Validation, Conversation Design, User Interface Design, UX Research, User Experience (UX), Team Leadership, Java, Web Design, Object-Oriented Programming (OOP), Data Collection, Data Structures, Algorithms, Programming, Microsoft Excel, HTML, Microsoft PowerPoint, Microsoft Office, C, python, java, JavaScript, HTML5, C (Programming Language), MySQL, Teamwork, Customer Service, Time Management, Leadership, communication, Collaborative Problem Solving, Spanish


========== P31 — position: dwelly ==========
HEADLINE: Product Engineer x AI/LLM Solutions Architect • building...
LOCATION: London, England, United Kingdom

SUMMARY:
Technical leader building AI systems that measurably improve investment and enterprise outcomes. I’ve led from 0→1→scale across VC/PE and regulated industries, shipping agentic workflows, LLM-powered analyses, and secure platforms in the cloud (AWS/GCP/Azure).

I pair hands-on engineering (Python/Node/React), data pipelines, and LLMOps/evals with product instincts and risk controls (SOC 2, PII, auditability).

My sweet spot: turning noisy workflows into reliable AI leverage with provable lift-time saved, higher throughput, and better decisions.

<personal>
My most-watched movie is Nefarious, my favourite fiction (?) book is Animal Farm, and my preferred artist is Braque. I keep a childlike spirit, finding joy in music, meaningful conversations, and travel.

I’m always eager to meet new people and learn new things. Feel free to reach out and share a story - after all, happiness is discovered through the voyage so one must imagine [io]sisyphus happy …
</personal>

EXPERIENCE:
- GenAI Advisor at Aeonic Software (2022-05 - present)
    Advisory role guiding product and engineering teams on GenAI application architecture and AI-powered product & strategy:
    • Advising on architecture and implementation of GenAI applications including RAG pipelines, vector
    search, agentic workflows, and LLM integration patterns
    • Guiding teams on best practices for evaluation frameworks, retrieval quality, and shipping reliable AI
    systems to production
    • Providing hands-on technical leadership on chunking & embedding strategies, agent
    orchestration for customer-facing AI products
- Product Engineer & Tech Advisor at LocalGlobe (2023-10 - 2025-09)
    Founding engineer on “Nazare,” the internal VC intelligence platform (sourcing, deal-flow triage, sentiment capture, collaboration, network etc.). Partnered with investment teams to encode real workflows; drove org-wide adoption from small pilot to daily use.
- Product Engineer & Operations Advisor at Advent (2023-01 - 2023-12)
    Spearheaded AdventGPT - Agentic LLM-powered internal tool focused on x10-ing PE professionals with sourcing, company analysis, IC papers, due diligence and other productivity tools.
    
    Built evaluation harnesses (precision/recall on RAG, answer-quality rubrics, hallucination flags) and entitlements.
- Technology Advisor (via YLD!) at Bulb (2022-05 - 2023-04)
    💡 was a wild rescue mission. I was hired to keep the lights on for the 1.5 million+ users during one of the biggest energy crises ⚡ Europe has ever seen (whilst most full-time engineers were abandoning a sinking 🚢). 
    
    Everything was event-driven, with microservices ⚙️ (some hundreds) in GCP with Node and some Python for data pipelines/ ingestion.  Their client interfaces were built with React and React Native, so I also had to maintain those from time to time. 
    
    Understanding and translating business flows into data flows whilst crucial technical founders were gone or transitioning was a real challenge - but every new grey hair added was worth it in the end. Met and collaborated with fantastic people and had a real sense of purpose given the macroeconomic and geopolitical environment. 🇺🇦
    
    The best thing is that I also helped Bulb's remaining team develop a POC (apps platform), which grew into Zoa (zoa.io - acquired by ENSEK).
- Chief Technology Officer at Eolas Medical (2021-04 - 2022-05)
    I have worn all the possible hats in a very intensive, hands-on, VC-funded early-stage start-up in a CTO/ Technical Founder role.
    
    Shipped white-label app generators (React Native & React) with 100% automated deployments; multi-tenant admin CMS. Serverless architecture on AWS (DynamoDB, Lambda, AppSync, API GW, SQS, Cognito, S3, OpenSearch, CloudFront, Route 53). 
    
    Assembled and harmonized functional dedicated web/ mobile/backend teams while focusing on nurturing culture. In the process, interviewed & hired cross functional teams and navigated through different working methodologies from Developer Anarchy, Kanban, and eventually Scrum.
    
    As can one imagine, med-tech is a highly regulated space so cybersecurity and massive attention to detail (as some parts of our product were even classified as medical devices) was crucial at all steps. Implemented all the security requirements to get the company SOC 2 compliant & pass pen tests.
- Senior Software Engineer at Relative (2019-02 - 2021-03)
    One of the biggest advantages of working within an agency is the multiple business domain exposure. During my time at Relative I got the chance to build from scratch (while architecting and even scaling some): 𝐇𝐨𝐦𝐞𝐒𝐞𝐫𝐯𝐞 𝐍𝐨𝐰 (React, React Native, Node, AWS), 𝐂𝐢𝐯𝐢𝐜 𝐃𝐨𝐥𝐥𝐚𝐫𝐬 (React Native, Node, AWS), 𝐑𝐞𝐬𝐭 𝐚𝐧𝐝 𝐁𝐞 (React Native, Ruby, AWS), 𝐀𝐩𝐩𝐂𝐡𝐞𝐟 (React, React Native, Node, Ruby, AWS), 𝐓𝐡𝐞 𝐏𝐢𝐧𝐭𝐞𝐫 (React Native, Node, AWS)
- Software Engineer at Thinslices (2018-04 - 2019-01)
    Thinslices is an agency providing software solutions for scale-ups and corporations. During my time at Thinslices, I got the chance to work with: 𝐑𝐨𝐥𝐚𝐧𝐝 𝐁𝐞𝐫𝐠𝐞𝐫 (React), 𝐇𝐢𝐧𝐝𝐚𝐰𝐢 (React).
- Software Engineer at Barukh Solutions (2016-07 - 2018-04)
    Umbrella company created mostly to sustain my expenses during uni & masters. Took gigs here and there, even failed a number of app SaaS start-ups (very early stage). 
    
    All in all a great period where tech was a safety net from my aspiration to change the world using political and social sciences. I realised fast enough that Plato got it all right and given my modest family background, the pragmatic choice for me as an aspiring free man (at least at that time) was the software engineering path. 
    
    Here is Plato’s quote:
    “And those who are to be the rulers must be the best of them all; hence they must be relieved from all other kinds of work, and dedicate themselves entirely to the business of ruling the state.”

EDUCATION:
- Utrecht University — Master's degree, International Development
- Universitatea „Alexandru Ioan Cuza” din Iași — Master's degree, International Development Studies in English
- Universitatea „Alexandru Ioan Cuza” din Iași — Bachelor's degree, Political Science

SKILLS:
(none)


========== P32 — position: dwelly ==========
HEADLINE: Graduate AI Engineer
LOCATION: United Kingdom

SUMMARY:
I’m also a freelance multi-disciplinary web designer and web developer who’s delivered creative and engaging solutions across digital media.


Skills: UI design, Web design, Web Application development, HTML5, CSS3, JavaScript, Bootstrap, AngularJS, NodeJS, ExpressJS, MongoDB, PHP, MySQL, C, SQL, Oracle, MEAN-stack development, full Stack web application development,MS.Office, Project Presentation, Project Documentation.

If you have a project I can help with, please get in touch.

EXPERIENCE:
- Full-stack web Developer at MicroSpark Software Solutions Private Limited (2017-01 - 2021-03)
    This is my new role. I'm feeling great to get this opportunity. I'll try my level best to serve this organisation with my skills.
- Frontend Web Developer at SSLABS (2015-12 - 2016-12)
    Full Stack Web Application Development
- Web Development Internship at SSLABS (2015-06 - 2015-11)
    flexible to acquire the challenges and endeavour to crack ASAP.

EDUCATION:
- University College Birmingham — Master of Science, Computer Science
- JNTUH College of Engineering Hyderabad — Bachelor's degree, Computer Science
- University of the West of Scotland — Postgraduate Diploma, Digital Marketing

SKILLS:
Computer Science, C, JavaScript, HTML5, Bootstrap, AngularJS, SQL, PHP, MySQL, Web Application Development, Web Application Design


========== P33 — position: dwelly ==========
HEADLINE: Developer Relations | Agentic AI Architecture | GTM | LLM Orchestration & Production Systems | London / EMEA
LOCATION: London Area, United Kingdom

SUMMARY:
I design AI systems that survive contact with production. I am a PhD-trained applied AI engineer specializing in LLM orchestration, agent frameworks, and Retrieval Augmented Generation (RAG). My work focuses on building systems that are reliable, maintainable, and architecturally sound.At Rasa, I architect advanced agentic AI curriculum and define production-ready reference architectures that guide enterprise deployments. I translate product capabilities into repeatable implementation patterns used by customers, partners, and system integrators.Previously, I built a biotech patent intelligence chat application in under three months using LangChain, LlamaIndex, AWS services, and OpenAI. I have deployed AI agent pipelines to thousands of learners via DataCamp and designed safeguard logic to reduce hallucination risk in LLM workflows.My background in cognitive neuroscience trained me to think in high-dimensional systems. Today, I apply that same rigor to production AI architecture.Core areas:• AI Agents (LangChain, LangGraph, Rasa, etc.)• Retrieval Augmented Generation• Vector databases• LLM deployment and orchestration• AWS-based AI systems• NLP pipelines and transformer modelsI work at the intersection of architecture, education, and enterprise AI adoption.

EXPERIENCE:
- Lead Curriculum Architect - Conversational AI / Agents at Rasa (2025-12 - present)
    Architectural ownership and platform adoption 
    
    - Defining repeatable, production-tested reference architectures that translate Rasa's conversational AI and agentic capabilities into real-world enterprise deployments
    
    - Acting as the architectural bridge between product engineering and enterprise adoption, ensuring system intent is preserved across documentation, education, and solution design
    
    - Shaping how customers, partners, and system integrators design, deploy, and evolve AI assistants in production environments
    
    - Architecting and leading Rasa's advanced agentic AI curriculum, focused on LLM orchestration, tool integration, system boundaries, and long-term maintainability
- Founder, DevRel / Applied AI at Genverv, Ltd. (2023-10 - present)
    I founded Genverv Ltd. to build production-grade generative AI systems for reasoning, retrieval, and action.
    
    I specialise in agentic systems, LLM orchestration, and semantic search, with deployments in biotech and education, including a patent-focused RAG system at Insmed and LangGraph agents used by 8,000+ learners on DataCamp.
    
    I create explainable, scalable AI tools, bridging engineering, product, and strategy. Open to partnerships and developer-facing roles in applied agentic AI.
- Subject Matter Expert (Applied AI) - LangGraph & AI Agents at DataCamp (2024-07 - 2025-01)
    Developing agentic workflows for automating tasks using LangChain/LangGraph.
- Generative AI Engineer & Consultant at Insmed Incorporated (2023-10 - 2024-01)
    Building Retrieval Augmented Generation (RAG) powered chatbot applications and customising large language models for bioinformatics/medtech use cases.
- NLP Engineering & Consultancy at  (2022-01 - 2023-10)
    Multiple projects involving an array of text analytics, including sentiment analysis for retail, text summarization, and text recommendation built on similarity.
- Machine Learning Web Apps at  (2019-04 - 2022-01)
    Machine learning web apps built using PyCaret and Streamlit, aimed at predicting behavioural outcomes across multiple settings.
- Machine Learning Specialist at Blicx Ltd (2021-07 - 2021-11)
    Provided data consultancy for video game research. Created machine-learning web app using PyCaret (Python machine-learning library) and Streamlit API to predict videogame recommendation using gaming attitudes survey. Commissioned by international game publisher via Blicx to model future gaming habits and ways to double market share.
- Postdoctoral Researcher for educational intervention, Centre for Brain and Cognitive Development at Birkbeck, University of London (2017-11 - 2019-04)
    Implemented educational intervention (randomised controlled trial) across 93 schools in England designed to facilitate math and science learning among primary school children. Managed UnLocke field agent activities to enable smooth classroom engagement with project software and supervised data collection across 6500 participating pupils. Conducted data analysis and collection for behavioural and neuroimaging components.
- PhD in Psychology (Cognitive Neuroscience with applied Machine Learning) at Bangor University (2013-10 - 2017-01)
    Used fMRI and machine-learning to identify patterns of brain activity related to sensorimotor experience during observational learning. Conducted 3 studies with adolescent and young adult subjects to investigate how visual and sensorimotor cortices indexed acquired whole-body movement experience during passive observation. All work has been published (see below).
- Research Coordinator - Laboratory of Motor Learning & Neural Plasticity at Concordia University (2011-01 - 2013-08)
    Data processing, recruiting, screening, and running participants for psychology/neuroscience projects featuring MRI and/or motion capture components.
- "A Facelift for Science Journalism" at  (2013-05 - 2013-05)
    Explores the pitfalls of science journalism today and proposes a new model for sharing science with the public. Highlights the importance of understanding basic research methods and statistical testing before distributing scientific information through media platforms.
- "What if we could control the brain?" at  (2012-11 - 2012-11)
    [link]
    
    Part of an independent TED Youth event entitled “The Power of Ideas”. Aimed at generating curiosity in neuroscience amongst younger audiences by highlighting the importance of recent advances in brain research.

EDUCATION:
- Bangor University — Doctor of Philosophy - PhD, Psychology (Neuroscience)
- Concordia University — Bachelor of Arts (BA), Psychology (Honours)

SKILLS:
Conversation Design, Solution Architecture, Business Intelligence (BI), [link], Retrieval-Augmented Generation (RAG), Large Language Models (LLM), Consulting, Research and Development (R&D), Amazon Textract, Vector databases, LangChain, Amazon Web Services (AWS), Transformers, Text Generation, Model Compression, AWS SageMaker, Probability, Text Classification, Sentiment Analysis, Named Entity Recognition (NER), Text Mining, BERT (Language Model), PyTorch, TensorFlow, Statistical Data Analysis, spaCy, HuggingFace, Natural Language Processing (NLP), Text Analytics, PyCaret, Pandas (Software), Streamlit, SPSS, Research, Psychology, Data Analysis, Neuroscience, Research Design, Statistics, Literature Reviews, Quantitative Research, Life Sciences, Project Management, Python (Programming Language), Public Speaking, Experimental Design, TEDx, Machine Learning, Deep Learning, Scikit-Learn


========== P34 — position: owner ==========
HEADLINE: Member of Technical Staff, Scaling at OpenAI
LOCATION: Palo Alto, United States

SUMMARY:
(none)

EXPERIENCE:
- Member of Technical Staff at OpenAI (2026-05 - present)
    Scaling at OpenAI
- Software Engineer at Meta (2025-09 - 2026-05)
    Meta Superintelligence Labs Infra
- Director of Engineering at Apple (2019-11 - 2025-09)
    Leads inference for Apple Foundation Models (AFM) and open source LLMs on public cloud at Apple. We run FMs at production scale across Apple products.
    
    Leads Apple Foundation Model pre-training data efforts, providing the tokens used by AFM team to pre-train their models.
    
    Leads Search Platform powering search across Apple products, including Siri, Safari, Apple Music, Apple TV, App Store, etc.
- Software Engineer at Google (2007-07 - 2016-05)
    Search Ranking, Search Infra, Search UI, Search Evaluation
- Tech Lead / Manager at Waymo (2016-05 - 2019-11)
    ML Training Infrastructure.  I built and tech lead the training infrastructure at Waymo, used by all teams to train perception, planning and other ML models at a large scale (2k+ TPUs).

EDUCATION:
- University of Toronto — Master of Science (M.Sc.), Computer Science (2005-01 - 2007-01)
- The University of British Columbia — Bachelor of Science (B.Sc.), Mathematics and Computer Science (2001-01 - 2005-01)

SKILLS:
MapReduce, Python, Machine Learning, Algorithms, C++, Distributed Systems, Java, Software Engineering, Computer Science, C, Algorithm Design, JavaScript, Programming, Tensorflow


========== P35 — position: dwelly ==========
HEADLINE: AI agents & Human-Machine interaction
LOCATION: Paris, Île-de-France, France

SUMMARY:
With over 10 years of hands-on experience as a CTPO in tech startups, I've been deep in the world of Human-Machine Interaction and AI (Robotics, Interactive 3D avatars, Mobile). Lately, my focus has shifted to AI agent frameworks, driven by the real-world challenges I’ve encountered in building intelligent, interactive and responsive systems.

EXPERIENCE:
- Product Engineer at Dust (2026-03 - present)
- Interim CTPO at Rizoa (2025-09 - 2026-03)
    Rizoa is an independent AI startup within the AFM (Association Famille Mulliez) ecosystem.
    
    Rizoa develops an AI copilot designed to support in-store employees of AFM retail brands in their day-to-day activities. The solution provides actionable recommendations and fast, intuitive access to relevant data, through a mobile-first web application tailored to the operational realities of frontline teams.
    
    𝗞𝗲𝘆 𝗰𝗼𝗻𝘁𝗿𝗶𝗯𝘂𝘁𝗶𝗼𝗻𝘀:
    • Product & tech vision: defined the product and technical vision to build a single, coherent platform able to address similar yet brand-specific needs across multiple retailers, relying on an agent-based architecture with dedicated business agents.
    • Team structuring: built, structured, and supported cross-functional product and engineering teams.
    • Product development: established core software development practices (CI/CD, testing, evaluation, ...) while remaining hands-on in the product’s development.
    
    𝗧𝗲𝗰𝗵 𝘀𝘁𝗮𝗰𝗸:
    • Frontend: React
    • Backend: Python, FastAPI, Chainlit
    • AI: LangChain, Text2SQL, RAG (QDrant), Langfuse
    • Infra: GCP
- Founder | AI agents & E2E test automation at TestAgent (2024-06 - 2025-06)
    I made engineering and product teams avoid costly bugs in their products by setting up in a week their end-to-end (E2E) test automation architecture and pipelines to run their main product flows.
    
    𝗔𝘀𝘀𝗶𝘀𝘁𝗶𝗻𝗴 𝗰𝗼𝗺𝗽𝗮𝗻𝗶𝗲𝘀 𝗶𝗻 𝘀𝗲𝘁𝘁𝗶𝗻𝗴 𝘂𝗽 𝘁𝗵𝗲𝗶𝗿 𝗘𝟮𝗘 𝘁𝗲𝘀𝘁 𝗮𝘂𝘁𝗼𝗺𝗮𝘁𝗶𝗼𝗻 𝗽𝗶𝗽𝗲𝗹𝗶𝗻𝗲 𝗳𝗼𝗿 𝗠𝗼𝗯𝗶𝗹𝗲, 𝗪𝗲𝗯 𝗮𝗻𝗱 𝗜𝗼𝗧 𝗮𝗽𝗽𝗹𝗶𝗰𝗮𝘁𝗶𝗼𝗻𝘀, 𝘄𝗶𝘁𝗵 𝗖𝗜/𝗖𝗗 𝗮𝗻𝗱 𝗹𝗲𝘃𝗲𝗿𝗮𝗴𝗶𝗻𝗴 𝗲𝘅𝗶𝘀𝘁𝗶𝗻𝗴 𝗳𝗿𝗮𝗺𝗲𝘄𝗼𝗿𝗸𝘀
    • Test automation frameworks
      ◦ Mobile: Appium, Waldo, Maestro (+ Browserstack, Saucelabs, BitBar, Perfecto, LambdaTest, ...)
      ◦ Web: Selenium, Cypress, Playwright
      ◦ IoT & connected devices: Robot Framework or concatenation of specific frameworks
    • Programming languages to fit best with their team skills: JavaScript, TypeScript, Python, Java, C#
    
    𝗘𝘅𝗽𝗹𝗼𝗿𝗶𝗻𝗴 𝘃𝗮𝗿𝗶𝗼𝘂𝘀 𝗽𝗿𝗼𝗷𝗲𝗰𝘁𝘀 𝗮𝗿𝗼𝘂𝗻𝗱 𝘁𝗲𝘀𝘁 𝗮𝘂𝘁𝗼𝗺𝗮𝘁𝗶𝗼𝗻 𝗮𝗻𝗱 𝗗𝗲𝘃𝗢𝗽𝘀:
    • full benchmark for existing solutions for test automation for Mobile/Web/IoT (tech stack exploration, features, pricing)
    • CI/CD pipeline with automated test on local MacOS runner (using fastlane connected to local MacOS gitlab runner to build a mobile application and run automated tests with Appium)
    • End to end (E2E) automated tests creation for various mobile applications using Appium + automation using device farm solutions (Browserstack, Saucelabs, BitBar, Perfecto, LambdaTest, ...)
    • ...
    
    𝗪𝗼𝗿𝗸𝗶𝗻𝗴 𝗼𝗻 𝗮 𝗻𝗲𝘄 𝘀𝗼𝗹𝘂𝘁𝗶𝗼𝗻 𝗳𝗼𝗿 𝗱𝘆𝗻𝗮𝗺𝗶𝗰 𝗴𝗲𝗻𝗲𝗿𝗮𝘁𝗶𝗼𝗻 𝗮𝗻𝗱 𝗺𝗮𝗶𝗻𝘁𝗲𝗻𝗮𝗻𝗰𝗲 𝗼𝗳 𝗘𝟮𝗘 𝗮𝘂𝘁𝗼𝗺𝗮𝘁𝗲𝗱 𝘁𝗲𝘀𝘁𝘀, 𝗹𝗲𝘃𝗲𝗿𝗮𝗴𝗶𝗻𝗴 𝗔𝗜 𝗮𝗴𝗲𝗻𝘁𝘀, 𝗛𝘂𝗺𝗮𝗻-𝗠𝗮𝗰𝗵𝗶𝗻𝗲-𝗜𝗻𝘁𝗲𝗿𝗮𝗰𝘁𝗶𝗼𝗻 𝗮𝗿𝗰𝗵𝗶𝘁𝗲𝗰𝘁𝘂𝗿𝗲 𝗮𝗻𝗱 𝗮𝗹𝗿𝗲𝗮𝗱𝘆 𝗲𝘅𝗶𝘀𝘁𝗶𝗻𝗴 𝘁𝗲𝘀𝘁 𝗳𝗿𝗮𝗺𝗲𝘄𝗼𝗿𝗸𝘀
- Tech & Product advisor at SPooN AI (2024-01 - 2025-06)
- Co-Founder & CT(P)O at SPooN AI (2016-07 - 2024-01)
    SPooN proposes an ecosystem (SDK & tools) to integrate 3D interactive characters in products (mobile apps, car assistants, websites, interactive totems) by merging Human-Machine Interaction, animation and AI.
    
    𝗞𝗲𝘆 𝗖𝗼𝗻𝘁𝗿𝗶𝗯𝘂𝘁𝗶𝗼𝗻𝘀:
    
    • 𝗖𝗼-𝗙𝗼𝘂𝗻𝗱𝗲𝗿:
      ◦ Helped define the business model and strategy.
      ◦ Contributed to fundraising efforts.
      ◦ Participated in driving company growth from inception to an ARR of multiple millions of euros.
    
    • 𝗖𝗧(𝗣)𝗢:
      ◦ Product development: contributed to product definition and roadmap.
      ◦ Technical leadership: directed and actively engaged in development (technical architecture definition, code review, mentoring)
      ◦ Project management: implemented agile methodologies to enhance internal processes and client deliveries.
      ◦ Team coordination: structured, expanded, and led the team from 3 to 20 members (including freelancers), hiring a successor to take over operational tasks, allowing me to concentrate on high-level strategy and long-term growth initiatives.
    
    • 𝗞𝗲𝘆 𝗮𝗰𝗰𝗼𝘂𝗻𝘁 𝗺𝗮𝗻𝗮𝗴𝗲𝗺𝗲𝗻𝘁:
      ◦ Managed the partnership with Renault, overseeing the deployment of the reno avatar in the Renault R5.
      ◦ Helped define the corresponding statement of work and business model.
      ◦ Structured operational teams to deliver innovative solutions on an accelerated timeline: from prototype to production in under two years.
      ◦ Built strong client relationships and collaborated closely on product strategy.
    
    𝗧𝗲𝗰𝗵 𝘀𝘁𝗮𝗰𝗸:
      ◦ Unity (C#)
      ◦ C++ modules for HMI technology integration
      ◦ Chatbots
      ◦ AI & LLM
      ◦ Native SDK for mobile and automotive OS: Kotlin (Android) & Swift (iOS)
      ◦ CI/CD (Gitlab CI) & Automated tests (Appium, Cucumber)
- Software Director at ALDEBARAN (definitely closed) (2012-04 - 2016-06)
    In charge of the Interaction SDK team (composed of 4 R&D Software teams, ~20 people).
    
    Jobs:
    • Perception Engineer (1,5 years)
    • High-level-integration Team Leader (9 months)
    • Interactivity Team Manager (1 year)
    • Software Director (1 year)

EDUCATION:
- Imperial College London — Master of Science (MSc), Computer Science (specialty in Artificial Intelligence)
- École Polytechnique — Computer Science & Electrical Engineering
- Lycée Sainte-Geneviève

SKILLS:
AI Agents, End-to-end Testing, Software Development, Internet of Things (IoT), Connected Devices, Mobile Applications, Cross-functional Team Leadership, Cross-Functional Team Building, TypeScript, Large Language Models (LLM), Human-robot Interaction, Robot Operating System (ROS), JavaScript, Test Automation, Test Automation Frameworks, Playwright, Cypress, Maestro, Waldo, Selenium


========== P36 — position: owner ==========
HEADLINE: Tech Lead, Meta Superintelligence Labs
LOCATION: New York, United States

SUMMARY:
HOW-TO<br>- Contact me about new roles: I'm not looking right now<br>- Ask for referral:…

EXPERIENCE:
- Tech Lead, Meta Superintelligence Labs at Meta (2020-01 - present)
    Tech Lead at Product & Applied Research, Meta Superintelligence Labs
- Software Development Engineer at Amazon (2016-01 - 2020-01)
    Amazon Alexa

EDUCATION:
- Drexel University — Computer Science

SKILLS:
Swift (Programming Language), Computer Science, iOS, Front-End Development, Distributed Systems, Django, FastAPI, JavaScript, Full-Stack Development, Hack (Programming Language), Deep Learning, Android, iOS Development, Kotlin, SwiftUI, Python (Programming Language), Machine Learning, Neural Networks, Natural Language Processing (NLP), Large Language Models (LLM)


========== P37 — position: owner ==========
HEADLINE: Google | Stony Brook University
LOCATION: United States

SUMMARY:
With over a decade of software engineering experience at industry leaders like Google and Microsoft, I specialize in architecting and building reliable distributed systems at massive scale. Currently, I help power Google Cloud's Zettabyte-scale storage infrastructure—supporting billions of users and mission-critical services around the world.

My expertise spans cloud computing, AI/ML, big data, and systems design, underpinned by a Master’s degree in Computer Science. I focus on designing and leading large-scale projects, with deep hands-on experience in machine learning, Natural Language Processing (NLP), and building scalable, high-impact solutions to complex real-world challenges.

I’m passionate about pushing the boundaries of what’s possible with technology, and I bring a thoughtful, systems-driven approach to everything I build.

Feel free to connect via [link] or email—I’m always open to meaningful conversations and collaborations.

EXPERIENCE:
- Advisory Board at University of California, Riverside (2021-12 - present)
- Microsoft Azure at Microsoft (2019-01 - present)
    Azure Intelligent Escort Team (Azure, C#, ML Studio, MapReduce, Geneva)
    ▪	Designing and developing session monitor solutions with services for activity detection, health monitoring, video creation, session-info search, etc.
    ▪	Implemented fault tolerance and high availability solutions for business continuity and disaster recovery.
    ▪	Designed and implemented services to monitor user sessions and extract machine learning insights like idle sessions, secret detections, labeling, etc.
    ▪	This solution resulted in increased productivity of DRI escorts by 100% while improving security by detecting security concerns.
    
    Cloud Operations Innovations Team (Azure, SQL, C#, D365, JavaScript)
    ▪	Design and develop a highly available, low latency system for Supply Chain.
- Google Cloud at Google (2024-01 - present)
- Software Engineer at Entain (2011-06 - 2014-03)
    • Designed and developed highly scalable, distributed and performant large-scale systems that handle millions of requests per day and also small tools and libraries that are used by multiple teams. 
    • Worked on integration of third-party games(web-services, APIs (RESTful/ Soap), games transactions system(3rd party/in-house), fault-tolerance system- (reconciliation system).
    • Collaborated and worked with various teams, 3rd party gaming-companies and regulatory bodies across the globe.
    • Implemented statistical and mathematical concepts to model various casino real-time, real-money games.
    • Designed and Developed (full-stack - server, JSP, database) various single-player and multiplayer casino games.
    ▪ Two of these games were ranked top 10 on the site.
    • Redesigned and developed casino games framework.
    • Worked on JDBC, web-services, APIs (RESTful/ Soap), JDBC, Oracle, Mapreduce.
- Grad School - MS Computer Science at Stony Brook University (2017-08 - 2019-01)
- Senior Software Engineer at Entain (2014-04 - 2017-07)
    • Designed and developed distributed systems to handle billions of requests, utilize and analyze TBs of data every day, coming from millions of users across 160+ countries and 35+ different brands.
    • Research and worked games personalization solutions that involved recommending better games to the users.
    • Lead software development team and did code and system design reviews.
    • Mentored and conducted tech sessions to promote learning in the company.
    • Applied and promoted an Agile Software development environment using Scrum and Kanban.

EDUCATION:
- National Institute of Technology Kurukshetra — Bachelor of Technology (B.Tech.), Computer Engineering
- Stony Brook University — Master's of Science, Computer Science (2017-01 - 2019-01)

SKILLS:
Artificial Intelligence (AI), Azure DevOps Services, Object-Oriented Programming (OOP), Hadoop, Oracle Database, .NET Framework, Azure Cosmos DB, ICM, Azure Service Bus, Microsoft Azure, Microsoft Azure Machine Learning, Data Science, Back-End Web Development, Java, Software Development, Agile & Waterfall Methodologies, Data Structures, Web Services, Algorithms, Design Patterns, Object Oriented Design, Machine Learning, JSP, Core Java, C, SQL, .NET, C++, PL/SQL, JavaScript, Servlets, Oracle, Eclipse, Spring, Java Enterprise Edition, Struts, Hibernate, TensorFlow, Scala, C#, MapReduce, Problem Solving, J2EE Application Development, Spark, Natural Language Processing, Algorithm Analysis, Python (Programming Language), Programming, Unix, MongoDB, Scrum


========== P38 — position: owner ==========
HEADLINE: Principal Software Engineer at SPECS | Founder of RecRoom | ex-HoloLens | ex-Xbox
LOCATION: Bothell, United States

SUMMARY:
I've spent the last two and a half decades working in the games industry, both on games themselves, and the platforms that support them.  As co-founder of Rec Room and architect of the cloud services that power it, I am passionate about the development of large-scale, high-concurrency services.  Prior to Rec Room, I was at Microsoft where I worked on the Xbox game console and on the HoloLens augmented reality device.

EXPERIENCE:
- Principal Software Engineer at SPECS (2026-04 - present)
- Co-Founder and Server Architect at Rec Room (2016-03 - 2026-04)
    Founding member of the company behind the hit social VR game Rec Room.
    • Senior leader of the engineering team
    • Architect of all back-end cloud services powering Rec Room, as well as many of the core systems within the game itself
- Senior Software Developer - HoloLens at Microsoft (2012-04 - 2016-03)
    Senior developer on the very first HoloLens app team responsible for multiple launch apps:
    • HoloStudio: The fast and simple workshop for building, printing and sharing your own holograms
    • RoboRaid: A mixed reality first person shooter where you defend your room from a robot invasion.
    • Built countless prototypes and demos that shaped the development of the HoloLens.
    • Developed the technology used in all HoloLens stage demos that allows the audience to see the same holograms that the presenter is seeing from dynamic 3rd person perspectives.
- Senior Software Developer - Xbox at Microsoft (2003-08 - 2012-04)
    Senior developer that worked on many facets of the Xbox operating system, contributing to the launch of the Xbox 360, Kinect, and Xbox One.

EDUCATION:
- Stanford University — BS, Computer Science (1998-01 - 2003-01)

SKILLS:
Virtual Reality, Augmented Reality, Mixed Reality, Software Engineering, Software Design, Software Development, Game Development, C#, C++, Agile Methodologies, Scrum, Visual Studio, Xbox 360, Agile Project Management, Software Project Management, Multithreading, Win32 API, Scalability, Test Automation, Distributed Systems, Testing, C, Debugging, Video Games, Unity3D, Networking, .NET Framework, Microsoft Azure, Oculus Rift, HTC Vive


========== P39 — position: owner ==========
HEADLINE: Engineering at Harvey | Track Day Enthusiast
LOCATION: San Francisco, United States

SUMMARY:
As a Software Engineer at Harvey, I contribute to the ongoing development and optimization of software solutions, leveraging skills in Python, React.js, and TypeScript. My work builds on experience collaborating with cross-functional teams in fast-paced environments to deliver high-quality results.  

I hold a Bachelor of Science in Computer Science from Purdue University, where I gained hands-on research and teaching experience. My professional journey includes roles at Aviatrix, where I played a key role in designing and building a next-generation SaaS application. My technical expertise reflects my commitment to innovation and precision in software engineering.

EXPERIENCE:
- Software Engineer at Harvey (2025-09 - present)
- Software Engineer Intern at Zotec Partners (2020-06 - 2020-12)
    - Developed a project to combat Covid-19 utilizing React, .NET Core, and DynamoDB.
    - Implemented a new module using AngularJs, Bootstrap, and SQL in an existing project.
    - Established several Python scripts to integrate two platforms.
    - Resolved Americans with Disabilities Act of 1990 violations in an existing project.
    - Evaluated and demonstrated new technology integration for the team.
- Teaching Assistant: Data Structures and Algorithms at Purdue University (2020-08 - 2021-12)
    - Collaborated with a team of faculty at meetings and actively contributed new ideas for labs and assignments.
    - Developed relationships with students, helped students to understand course materials better and to finish their tasks.
    - Mentored students one-on-one during office hours and labs to address each individual’s questions and concerns and provided tips to help them succeed in the Computer Science field.
- Senior Software Engineer at Aviatrix (2024-01 - 2025-07)
    Lead full-stack engineer with a focus on UI for Aviatrix’s next-generation SaaS offering (PaaS). Designed and built the application from the ground up, deploying it to production within one year. Collaborate closely with backend engineers, PMs, and UX designers in a fast-paced startup environment. Tech stack includes React, TypeScript, Go, gRPC-web, gRPC, ProtoBuf, Postgres, K8s and more.
- Teaching Assistant: Programming in C at Purdue University (2019-08 - 2020-05)
    - Collaborated with a team of faculty at weekly meetings and actively contributed new ideas for labs and assignments.
    - Developed relationships with students, helped students to understand course materials better and to finish their tasks.
    - Mentored students one-on-one during office hours and labs to address each individual’s questions and concerns and provided tips to help them succeed in the Computer Science field.
    - Lead students during lab sessions to make sure each individual can apply their knowledge to real coding challenges.
- Software Engineer at Aviatrix (2022-07 - 2024-01)
- Undergraduate Researcher at Purdue University (2020-04 - 2022-05)
    - Performed research in theoretical chemistry using Java, JavaFx, and Jmol. 
    - Worked with a group of researchers within the Slipchenko Lab Group at Purdue University.
- Personal goal pursuit at Career Break (2025-07 - 2025-09)
    Immigration status transition

EDUCATION:
- Purdue University — Bachelor of Science - BS, Computer Science (2018-08 - 2022-05)

SKILLS:
Python (Programming Language), React.js, TypeScript, Go, gRPC, gRPC-web, Protocol Buffers, JavaScript, D3.js, Node.js, HTML, SQLite, Jira, Agile Methodologies, Scrum, Git, Elasticsearch, REST APIs, Java, C (Programming Language), SQL, Linux, Leadership, Time Management, Accountability, Bash, Cascading Style Sheets (CSS)


========== P40 — position: owner ==========
HEADLINE: Software Engineer at Luma AI
LOCATION: San Francisco, United States

SUMMARY:
(none)

EXPERIENCE:
- Software Engineer at Luma AI (2025-06 - present)
- Senior Software Engineer at NVIDIA (2019-10 - 2020-05)
    Member of the Geforce Now Core team (Cloud Gaming)
    
    Co-tech lead of Reverseproxy and Loadbalancer microservice. 
    
    Responsible for directing traffic to internal microservices, minimizing latency and packet drops for packets flowing in and out of Nvidia datacenters.
    
    Technologies used: HAProxy, NS1, Iptables, Python, Ubuntu, CentOS, Firebase, Mellanox NIC, Intel NIC, Ethtool
- Intern at Smart Applications on Virtual Infrastructure (SAVI) at University of Toronto (2014-05 - 2014-07)
    • Designed and implemented a software defined network overlay which enabled communications for user defined network topologies spread across Canada.
    
    • Wrote automated jobs which were used by several teams to help speed up manual tasks. 
    
    [link]
- Software Engineer at NVIDIA (2017-01 - 2019-10)
    Scaled Cassandra database to handle 150 million requests a week. Worked closely with application side queries and with the C# driver to reduce latencies and timeouts. 
    
    Took ownership of autoscaler code that dynamically spawned AWS resources when Nvidia datacenters reached peak capacity. 
    
    Technologies used: AWS ec2, AWS s3, AWS codedeploy, AWS lambda, AWS ASG, AWS SQS, Elasticsearch, Kibana, Watchers, Datastax Cassandra C# driver, C#, Windows servers
- Undergraduate Research Assistant for Smart Applications on Virtual Infrastructure (SAVI) at University of Toronto (2013-07 - 2013-09)
    • Deployed Nagios to setup real-time monitoring for cloud infrastructure across Canada with the use of an app and website.
- Software Engineering Intern at NVIDIA (2016-05 - 2016-12)
    Member of the GeForce NOW Core team (Cloud Gaming)
    
    • Implemented automated server side code deployments to a hybrid cloud infrastructure using AWS Codedeploy and Jenkins. Reduced deployment time by 300%.
    
    • Added Kibana dashboards to track real time metrics such as resource availability and API call latencies. 
    
    • Used metrics, watchers and logs to help find and fix production bugs.  
- Senior Software Engineer at Twitch (2020-05 - 2025-06)
- Software Engineering Intern at Microsemi Corporation (2014-07 - 2015-08)
    Member of Libero SoC Application Frameworks and Infrastructure team
     
    • Designed and implemented a modularized code base for the next generation Physical Design Constraint editor. 
    
    • Reduced overall code by 17% while adding an improved UI, logging and debug messaging. 
    
    • Enhanced the New Project Wizard user interface to allow selecting FPGAs from a dynamic table. 

EDUCATION:
- University of Toronto — Bachelor of Applied Science (B.A.Sc.) with Honours, Electrical and Electronics Engineering (2011-01 - 2016-01)

SKILLS:
C, C++, Python, Matlab, Verilog, Java, HTML, Assembly Language, Object Oriented Design, Algorithm Design, Data Structures, JavaScript, VHDL, Compilers
