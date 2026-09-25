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

## Brief: Autofleet (profiles under "autofleet")
Role: Senior Full Stack Developer at Autofleet (fleet management software), Tel Aviv, hybrid.
Must: Hands-on Node.js backend experience (Express or similar) / Hands-on React frontend experience / Minimum 5 years of hands-on software development / Based in Israel, within reach of Tel Aviv.
Nice: PostgreSQL or MongoDB / AWS / Currently at a strong Israeli product company (Wix, Monday, Fiverr, Similarweb level) / Degree from a top Israeli university (Technion, Tel Aviv University, Hebrew University, Weizmann, Ben-Gurion, Bar-Ilan) / 2+ years at current company.
Exclude: Job hopper: several roles under 1 year each / Career mostly at consulting, outsourcing or body-shop companies / Not hands-on any more (pure manager).

## Brief: ScaleOps (profiles under "scaleops")
Role: Backend Engineer at ScaleOps (real-time automated Kubernetes resource management and cloud cost optimization), hands-on individual contributor, Tel Aviv office 5 days a week.
Must: General backend (or backend-leaning full-stack) engineering is the person's actual day-to-day work / 4 to 10 years of hands-on software engineering / Lives within commuting range of Tel Aviv (a bare "Israel" with no city is fine; Haifa, Jerusalem, Be'er Sheva, the North or South, or abroad is not) / At least one excellence signal: a strong product company, an elite army tech unit, or a degree from a top university.
Nice: Kubernetes, distributed systems or infrastructure work / Go or Node.js backend / Currently at a strong Israeli product company.
Exclude: Specialist career in security research, machine learning, data science or data engineering / Frontend-leaning full-stack / QA-only, DevOps-only or automation-only background / Consulting, freelance or founder-on-the-side headline / More than 10 years of engineering (overqualified).

========== Q01 — position: scaleops ==========
ID: Q01
HEADLINE: Senior Software Engineer at Ownera
LOCATION: Tel-Aviv, Israel

SUMMARY:
(none)

EXPERIENCE:
- Co Founder, CTO at Viaphone Payments LLC (2015-11 - present)
- Senior Software Engineer at Ownera (2021-01 - present)
- Head Of Trading System Development at Kazakhstan stock exchange (2014-09 - 2016-06)
    In 2013, the exchange has matured to the development of a new trading system, which was supposed to update the old one in C++, written in the late 90s to a new one written in Java. I was appointed to lead the development of the project. In the course of the project, I was also an architect and developed several main modules of the system.
    
    Java is not often used in such low-latency systems, but I was able to find a great example from the guys from LMAX who just launched a similar system in Java and perfectly described the whole architecture: [link]
    
    For two years, we have been able to develop all the main modules of the system: the kernel, the FIX gateway, the JavaFX trading terminal, the fast Oracle DB client, the administrative module and module that transmits real-time bidding data to Bloomberg and Returer.
    
    A temporary gateway connecting the old and the new system has also been implemented. The module allowed to launch the entire system seamlessly and in stages, for half a year the data were duplicated between the two systems, and every month we included a new module in production. The final transition was completely unnoticed by the bidders, because by this time we just behind the scenes switched gateways.
    
    The system was launched in the middle of the 15th year, at the moment through it tens of thousands of transactions for many billions of dollars were carried out
    
    In 2014 I became the head of the development of trading systems. Under my wing got 7 developers and several testers, as well as a dozen real-time systems - core exchange systems. Together with the team, we have significantly improved our workflows, implemented quick task setting, automated testing, сontinuous integration, and continuous deployment. This allowed the unit to develop and efficiently launch a record number of updates and new modules, despite all the bureaucracy in the company.
- Lead Specialist at Kazakhstan Stock Exchange (2011-03 - 2014-09)
    Immediately after I got on the stock exchange, I was thrown on the launch of the trading platform for brokers STrade, where I replaced the two previous developers and launched the project.
    
    The platform at that time consisted of:
    Trade terminal built on Java Swing
    Java-server communicating with the terminal with protobuf-messages on top of the socket connection
    C++ gateway communicating with the core trading system, also written in C++
    
    In the process of launching the project, I implemented an adapter  that allows to connect any java-application to the trading system. The adapter allowed to exclude from the scheme an extra c++ gateway that occupied a huge amount of developers time. In fact, the adapter allowed to run the entire solution faster. It was was written outside the management plan, in the evenings outside of work.
    
    Subsequently, java-adapter allowed developing several more important exchange systems, connecting it via FIX API with brokers, banks and international systems.
    
    Became the head of this direction, two more developers and a tester were connected to the project.
- Java Developer at SmartPhone Labs LLC (2006-03 - 2007-08)
    Developed a wrapper for j2me-applications, which allows to control the work of the midlet, limit the application's work in time and add any extra menu. The utility made it possible to make a trial version of any mobile game on the fly and did not require the source code of the program. It was part of the wap-portal selling mobile games and was considered by EA for purchase.
- Head Of Software Development at Universal.kz (2007-09 - 2011-03)
    Developed from scratch wap-portal for the sale of mobile games. Built on EJB 3, Jboss Seam and Wurfl. The portal included a catalog of several hundred top games, was able to recognize the mobile device and select the necessary game distribution, graphics and content for it. It was also connected to two payment systems, the operator’s backbone, via paid sms and paid wap-links. Included analytical and content modules. It sold several tens of thousands of games, which allowed the company to exist for several years.
    
    Developed SMS bots that implement quizzes, chats on the TV channel, additional services, sale of music and content. Media solutions at that time had a real mass user throughout the country and brought a record income in terms of the number of developers.
    
    Led a group of several developers. This was my first leadership experience.
- Founder and CTO Aitec.one at Aitec.one (2017-09 - 2021-01)

EDUCATION:
- Gymnasium # 134 of the city of Almaty (1999-01 - 2002-01)
- Saint Petersburg State University — Master’s Degree, Applied Mathematics, Software Engineering (2002-01 - 2007-01)

SKILLS:
Software Development, Mobile Applications, Java, Subversion, SQL, jQuery, Git, XML, iOS development, Hibernate, Visual Studio, Web Services, Linux, JDBC, Testing, C#, JUnit, OOP, Programming, C, Java Enterprise Edition, MySQL, Python, PL/SQL, PHP, Ant, Agile Methodologies, T-SQL, Web Development, Eclipse, Scrum, C++, JavaScript, HTML, Maven, Android, Software Project Management

========== Q02 — position: scaleops ==========
ID: Q02
HEADLINE: SW & Developer Experience
LOCATION: Tel Aviv-Yafo, Israel

SUMMARY:
DevOps & Data Platform Engineer | Specializing in K8S, AWS, Solution Architecting and Data Lakehouse Design & Maintenance
Passionate & experienced DevOps Engineer, with a varied experience
in many fields, including on-premise DevOps practices, IT Skills, Public
Cloud Administration and Big Data Services.
High Interpersonal skills, with past experience in instructing, rebuilding
and restructuring training programs & exercises - all done with utmost
devotion for the cause. Looking for my next challenge in the DevOps
industry!

EXPERIENCE:
- Software Engineer, Developer Experience at Forter (2025-03 - present)
- DevOps and Data Platform Engineer at Israel Defense Forces - Unit Matzov (2023-08 - 2024-05)
- DevOps Course Instructor, DevOps Engineer at Basmach - IDF School of Computer Science (2021-08 - 2023-08)
- DevOps Lead at LightSolver (2024-06 - 2025-03)
    Leading the DevOps, Platform Engineering, Data Engineering and IT endeavors in the company

EDUCATION:
- The Open University of Israel — Bachelor's degree, Computer Science and Earth Sciences (Double Major) (2024-10 - 2029-08)
- Basmach - IDF School of Computer Science — Technical Course, DevOps Engineering (2021-03 - 2021-07)
- Yehud Comprehensive High School — High School Diploma, Cyber Security & Physics (2016-09 - 2019-05)
- Future Scientists Center (2016-09 - 2019-05)

SKILLS:
Python (Programming Language), Continuous Integration and Continuous Delivery (CI/CD), Amazon Web Services (AWS), [link], Ethnography, JavaScript, Jenkins, GNU Make, [link] Copilot, Problem Solving, Communication, Helm (Software), System Architecture, Solution Architecture, Scripting, Linux System Administration, Terraform, OpenShift, Apache Spark, Apache Kafka

========== Q03 — position: autofleet ==========
ID: Q03
HEADLINE: Full Stack Developer at Hello Heart
LOCATION: Israel

SUMMARY:
Hello Heart's mission is to empower people to understand and improve their health using smartphone technology. We help employers and health plans take heart risks under control with a clinically-based smartphone solution

EXPERIENCE:
- Senior Full Stack Developer at Hello Heart (2023-01 - present)
- Java Software Developer at IAF - Israeli Air Force (2016-03 - 2017-12)
- Full Stack Engineer at Omnistream (2019-03 - 2021-02)
- Frontend Developer at IAF - Israeli Air Force (2017-12 - 2019-03)
- Frontend Developer at Glassbox (2021-02 - 2023-01)

EDUCATION:
- The College of Management Academic Studies — Computer Science (2017-01 - 2020-01)
- Basmach - Mamram — Computer Science (2015-01 - 2016-01)

SKILLS:
Microsoft Office, Java, Software Development, HTML, JavaScript, Object-Oriented Programming (OOP), Cascading Style Sheets (CSS), Android

========== Q04 — position: scaleops ==========
ID: Q04
HEADLINE: Full Stack Developer at Hello Heart
LOCATION: Israel

SUMMARY:
Hello Heart's mission is to empower people to understand and improve their health using smartphone technology. We help employers and health plans take heart risks under control with a clinically-based smartphone solution

EXPERIENCE:
- Senior Full Stack Developer at Hello Heart (2023-01 - present)
- Java Software Developer at IAF - Israeli Air Force (2016-03 - 2017-12)
- Full Stack Engineer at Omnistream (2019-03 - 2021-02)
- Frontend Developer at IAF - Israeli Air Force (2017-12 - 2019-03)
- Frontend Developer at Glassbox (2021-02 - 2023-01)

EDUCATION:
- The College of Management Academic Studies — Computer Science (2017-01 - 2020-01)
- Basmach - Mamram — Computer Science (2015-01 - 2016-01)

SKILLS:
Microsoft Office, Java, Software Development, HTML, JavaScript, Object-Oriented Programming (OOP), Cascading Style Sheets (CSS), Android

========== Q05 — position: autofleet ==========
ID: Q05
HEADLINE: -
LOCATION: Israel

SUMMARY:
(none)

EXPERIENCE:
- Full-stack Developer at SAP (2021-12 - present)
- Full Stack Engineer at NGSoft Ltd. BATM Group (2019-12 - 2021-10)

EDUCATION:
- Ben-Gurion University of the Negev — Software and Information Systems Engineering (2015-01 - 2019-01)

SKILLS:
java, data sciense, Python (Programming Language), AngularJS, MongoDB, cloudant, Artificial Intelligence (AI), JavaScript, Node.js, HTML, Cascading Style Sheets (CSS), Flask, SQL, React.js, TypeScript

========== Q06 — position: scaleops ==========
ID: Q06
HEADLINE: Backend Software Team Leader at Claroty
LOCATION: Tel Aviv District, Israel

SUMMARY:
An Engineering Team Leader and a former developer with 7 years of experience.

• Fluent and experienced in Python language.
• Wide knowledge of Cybersecurity and Networking.
• Familiar with many Devops technologies as Docker, Jenkins, Openshift etc.

EXPERIENCE:
- Backend Software Team Leader at Claroty (2025-03 - present)
    • Collection Team Lead.
    • Built and leading a high-performing team, overseeing hiring, onboarding, and team development.
    • Defined and executing short and long-term technical vision and strategy to align with organizational goals.
    • Leading projects using Python and C++ as core technology stacks, ensuring robust, scalable solutions.
- Backend Software Engineer at Claroty (2022-11 - 2025-03)
    • Develop and Refactor major components of CTD - on premise threat detection system.
    • Organizer of R&D’s weekly tech events and responsible for the team’s onboarding process.
    • Managed projects and tasks of our visibility team.
    • Large impact on the system’s architecture plan.
- Backend Software Engineer at … (2020-10 - 2022-10)
    • Developed a cyber defence research tools, frameworks and applications.
    • 2021 R&D’s outstanding worker.
- Cyber Security Researcher at … (2019-02 - 2020-10)
    • Conducted intricate cyber threat research.

EDUCATION:
- The Open University of Israel — Bachelor of Arts - BA, Economics and Management (2024-10)

SKILLS:
Python (Programming Language), Team Leadership, Management, Networking, Software Design, Dockers, Code Review, Leadership, English, Object-Oriented Programming (OOP), Git, Debugging, Scale, C++, React.js, Communication, Agile, System Architecture, Security, Continuous Threat Detection (CTD)

========== Q07 — position: autofleet ==========
ID: Q07
HEADLINE: Full stack software engineer at Akamai Technologies
LOCATION: Israel

SUMMARY:
Dedicated and efficient Full Stack Developer with 8 years of experience in web applications and databases. Highly motivated, team-oriented, and capable of excelling under pressure. Certified in both Frontend and Backend technologies, demonstrating a commitment to continuous learning. Skilled in Vue, React, Angular, Node.js, TypeScript, and various databases.
With a Bachelor of Science (BSc) focused in Information Systems from The Academic College of Tel-Aviv, Yaffo.

EXPERIENCE:
- Full stack software engineer at Akamai Technologies (2024-02 - present)
- Full-stack Developer at Novelty Media (2019-07 - 2023-01)
- Frontend Developer at OFAKIM Group (2017-08 - 2019-07)
- Software Developer at Sapiens (2016-07 - 2017-08)
    Development of web information system for Tel Aviv Municipality.  
    •	Writing complex sql queries.
    •	Implement UI design using CSS3 and java Script.
    •	Being a team member from scratch.
- Full-stack Developer at Decido (2023-02 - 2023-11)

EDUCATION:
- The Academic College of Tel-Aviv, Yaffo — Bachelor of Science (BSc), Management Information Systems (2013-10 - 2016-07)
- Ohel-Shem High School

SKILLS:
Debugging Code, Technical Solution Design, Debugging, Technical Design, React Native, Server Side, Framework Design, Skill Development, Scripting, API Development, Vue.js, Vuex, Vue, Microsoft SQL Server, Google Cloud Platform (GCP), Bootstrap, Responsive Web Design, Representational State Transfer (REST), Git, Web Development

========== Q08 — position: scaleops ==========
ID: Q08
HEADLINE: Software Engineer at Monday.com
LOCATION: Israel

SUMMARY:
Experienced software developer, 9900 alumni, beginning to study in the Technion.
 4 Years experience with c#, node.js, c++, javascript, html and css.
Managed development and integration projects in the army.
Work well in teams and alone.

EXPERIENCE:
- Software Engineer at monday.com (2024-10 - present)
- VP R&D at Gist MD (2020-10 - 2025-01)
- Full Stack Engineer at Gist MD (2020-06 - 2025-01)
- Backend Developer at Unit 9900 - Israeli Intelligence Corps (2017-01 - 2019-09)

EDUCATION:
- Technion - Israel Institute of Technology — Computer Software Engineering (2019-01 - 2023-01)

SKILLS:
React.js, Programming, System Performance, Software Systems, Technical Leadership, Communication, Code Review, Defining Requirements, Skill Development, Team Management, Team Development, Software Development, Full-Stack Development, Node.js, C#, C++, Web Development, Linux, ASP.NET, Decision-Making

========== Q09 — position: autofleet ==========
ID: Q09
HEADLINE: Full Stack Developer at Tonkean
LOCATION: Israel

SUMMARY:
Graduated B.Sc in Physics and Computer Science at The Hebrew University of Jerusalem (2017).
Skilled programmer, self learner and passionate about cutting-edge technologies,
Looking for the next challenge!

EXPERIENCE:
- Full-stack Developer at Tonkean (2024-01 - 2025-10)
- Full Stack Engineer at Amenity Analytics (2021-02 - 2023-08)
    React, Typescript, AWS, Python, SQL
- Software Engineer at Taboola (2019-06 - 2021-02)
    Java, SQL, Big Query, RTB, React, SCSS
- Full Stack Developer at Yellow Pages Israel (2019-04 - 2019-05)
    C#, SQL Server
- React Native Developer at Yellow Pages Israel (2018-02 - 2019-04)
    Rewrote the "Zap Price Comparison" app in React Native together with the mobile team.
    React Native, Redux
- Android Developer at 200apps (2016-10 - 2018-02)
    - Native Android app development: custom UI components, logic, API's requests.
    - Developed android application from scratch with: RxJava, Dagger2, MVP concept, GSON and more.
    - Working individually, with a team of android developers, and with designers.
- Program developer in a physics lab at The Hebrew University of Jerusalem (2015-09 - 2016-10)
    - Tool development for lab experiments using Arduino.
    - GUI writing and tool development in Matlab.
- QA Engineer at Genie Solutions (2009-01 - 2010-01)

EDUCATION:
- The Hebrew University of Jerusalem — Bachelor of Science (B.Sc.), Computer Science and Physics (2014-01 - 2017-01)

SKILLS:
React, Kotlin, AWS, TypeScript, JavaScript, React.js, Java, Android Development, React Native, Object-Oriented Programming (OOP), Image Processing, Physics, Python, Matlab, SQL, C#

========== Q10 — position: scaleops ==========
ID: Q10
HEADLINE: Software Engineer at VAST Data
LOCATION: Tel Aviv District, Israel

SUMMARY:
A software engineer at Vast Data, Computer Science graduate, Technion. Knowledge in objective C, C++, and Python.  Skilled in teamwork and leadership.

EXPERIENCE:
- Software Engineer at VAST Data (2021-09 - present)
- Control System Manager at SCD (2017-01 - 2019-08)
- Company Commander at Israel Defense Forces (2015-03 - 2016-07)
    Commanded 150 soldiers. Including training phase and Operational activity.
- Software Engineer at Amazon Web Services (2019-08 - 2021-09)
- Team Leader at Israel Defense Forces (2010-03 - 2015-03)

EDUCATION:
- Technion - Israel Institute of Technology — Bachelor's degree, Computer Science (2017-01 - 2020-01)

SKILLS:
Leadership, C++, c, Linux, c++, bash, Team Management, Teamwork, Public Speaking, Self Learning, C (Programming Language), Python (Programming Language), Dart, Flutter

========== Q11 — position: autofleet ==========
ID: Q11
HEADLINE: Senior Full Stack Developer at Minute Media
LOCATION: Israel

SUMMARY:
Bachelor of Computer Science, Technion – Israel Institute of Technology. Eager to take part in the development of future leading technologies, able to grasp new concepts quickly and efficiently, ever striving to acquire and master new skills.

EXPERIENCE:
- Senior Full Stack Developer at 90min (2026-06 - present)
- Full Stack Developer at 90min (2022-07 - present)
- Frontend Developer at 90min (2021-06 - 2022-07)
- Android App with Angular 8 Management Interface – SeatMe at Technion - Israel Institute of Technology (2019-06 - 2020-04)
    The app is being used today by Technion students who use the Computer Science faculty library, the Angular app is being used by the librarians:
    [link]
    
    SeatMe was developed as one of my final projects of my Technion degree:
    • An Android based application and an Angular 8 web application for managing academic libraries and learning spaces, designed to allow real-time management and resources utilization
    • I led a research to understand the needs of the librarians and students, by interviewing them thoroughly, and to convince the CS faculty management, that this is worth pursing and to officially use it once done, which included several meetings that led to adopting the app
    • The work on the project was done by a team of 3, using Android Libraries, Angular 8 web development, Firebase services and NFC technologies. The development process was done on [link], using the Gitflow workflow, using the Agile project management with Scrum
- React Web Application – Traveling Social Network at Technion - Israel Institute of Technology (2019-12 - 2020-01)
    • A traveling social network, in which users create an account, follow friends, post their travel plans and plan them with friends
    • The app included a Flask backend and a React frontend, with REST APIs and a PostgreSQL database
    • User and password authentication, with authorization permissions according the users' roles
    • A map showing friends’ travel plans, subscribing to friends’ posts and getting notified whenever they are updated

EDUCATION:
- Technion - Israel Institute of Technology — Bachelor's degree, Computer Science
- Technion - Israel Institute of Technology — Computer Science (2015-01 - 2020-01)

SKILLS:
Go (Programming Language), JavaScript

========== Q12 — position: scaleops ==========
ID: Q12
HEADLINE: Senior Full Stack Developer at Minute Media
LOCATION: Israel

SUMMARY:
Bachelor of Computer Science, Technion – Israel Institute of Technology. Eager to take part in the development of future leading technologies, able to grasp new concepts quickly and efficiently, ever striving to acquire and master new skills.

EXPERIENCE:
- Senior Full Stack Developer at 90min (2026-06 - present)
- Full Stack Developer at 90min (2022-07 - present)
- Frontend Developer at 90min (2021-06 - 2022-07)
- Android App with Angular 8 Management Interface – SeatMe at Technion - Israel Institute of Technology (2019-06 - 2020-04)
    The app is being used today by Technion students who use the Computer Science faculty library, the Angular app is being used by the librarians:
    [link]
    
    SeatMe was developed as one of my final projects of my Technion degree:
    • An Android based application and an Angular 8 web application for managing academic libraries and learning spaces, designed to allow real-time management and resources utilization
    • I led a research to understand the needs of the librarians and students, by interviewing them thoroughly, and to convince the CS faculty management, that this is worth pursing and to officially use it once done, which included several meetings that led to adopting the app
    • The work on the project was done by a team of 3, using Android Libraries, Angular 8 web development, Firebase services and NFC technologies. The development process was done on [link], using the Gitflow workflow, using the Agile project management with Scrum
- React Web Application – Traveling Social Network at Technion - Israel Institute of Technology (2019-12 - 2020-01)
    • A traveling social network, in which users create an account, follow friends, post their travel plans and plan them with friends
    • The app included a Flask backend and a React frontend, with REST APIs and a PostgreSQL database
    • User and password authentication, with authorization permissions according the users' roles
    • A map showing friends’ travel plans, subscribing to friends’ posts and getting notified whenever they are updated

EDUCATION:
- Technion - Israel Institute of Technology — Bachelor's degree, Computer Science
- Technion - Israel Institute of Technology — Computer Science (2015-01 - 2020-01)

SKILLS:
Go (Programming Language), JavaScript

========== Q13 — position: scaleops ==========
ID: Q13
HEADLINE: Headline because it is required
LOCATION: Israel

SUMMARY:
(none)

EXPERIENCE:
- Principal Engineer at Palo Alto Networks (2022-02 - present)
- Software Developer at Check Point Software Technologies (2012-07 - 2013-10)
- Strategic research at AppDome (2013-09 - 2022-01)

EDUCATION:
- Technion - Israel Institute of Technology — Bachelor of Science (BS), Computer Science (2007-01 - 2013-01)

SKILLS:
Linux, Network Security, Object Oriented Design, TCP/IP, Java, Multithreading, Software Development, C++, Computer Security, Perl, Bash, System Architecture, Shell Scripting, Operating Systems, VPN

========== Q14 — position: scaleops ==========
ID: Q14
HEADLINE: Software Architect at Palo Alto Networks
LOCATION: Rishon LeZion, Israel

SUMMARY:
(none)

EXPERIENCE:
- Software Architect at Palo Alto Networks (2026-02 - present)
- Software Architect at CyberArk (2026-01 - present)
- Software Engineer at CyberArk (2021-04 - present)
- Software Development Team Lead at Honeywell Process Solutions (2019-04 - 2021-04)
- Software Developer at Honeywell Process Solutions (2017-09 - 2021-04)
- Software Developer at NextNine (2016-01 - 2021-04)

EDUCATION:
- ORT Colleges — Practical Software Engineer, Computer Software Engineering (2010-01 - 2012-01)
- The Academic College of Tel-Aviv, Yaffo — Bachelor's degree, Computer Science (2015-01 - 2019-01)

SKILLS:
Perl, Python, Java, VBScript, Powershell, Software Development, Customer Support, Customer Service

========== Q15 — position: autofleet ==========
ID: Q15
HEADLINE: Full Stack Developer at Tabit - Restaurant Technologies
LOCATION: Israel

SUMMARY:
(none)

EXPERIENCE:
- Full Stack Developer at Tabit - Restaurant Technologies (2020-07 - present)
- Full Stack Web Developer at Paragon Ltd. (2020-01 - 2020-07)
- Network Operations Center Operator at Bank of Israel (2018-12 - 2019-12)

EDUCATION:
- Jerusalem College of Engineering — Bachelor of Science - BS, Computer Software Engineering (2014-01 - 2019-01)

SKILLS:
JavaScript, React.js, Node.js, Object-Oriented Programming (OOP), Test Driven Development, Cascading Style Sheets (CSS), HTML5, C, Java, Python, SQL, Redux.js, SASS, MongoDB, Bootstrap, jQuery, AngularJS, TypeScript, npm, Express.js, angular

========== Q16 — position: scaleops ==========
ID: Q16
HEADLINE: Senior Business Data Analytics Team Lead at monday.com
LOCATION: Israel

SUMMARY:
(none)

EXPERIENCE:
- Senior Customer Organization Analytics Team Lead at monday.com (2026-08 - present)
    GTM Strategy & Operations Group
- Customer Organization Analytics Team Lead at monday.com (2025-09 - 2026-08)
- Customer Experience Analytics Team Lead at monday.com (2025-04 - 2025-09)
- Senior Business Data Analyst at monday.com (2024-09 - 2025-04)
- Business Data Analyst at monday.com (2023-07 - 2024-09)
- Business Analyst at Perimeter 81, a Check Point Company (2022-06 - 2023-07)
- Data Analyst at Forescout Technologies Inc. (2021-01 - 2022-06)
    Product Operations Analytics Team -> Data Classification Team
- Quality Assurance Analyst at Artimedia (2018-04 - 2019-02)
- Data Analytics Team Leader at Israeli Military Intelligence - Unit 8200 (2016-10 - 2017-10)
    Team management of 12 data analysts and intelligence system operators
- Data Analyst at Israeli Military Intelligence - Unit 8200 (2015-07 - 2017-10)

EDUCATION:
- Shenkar - Engineering. Design. Art. — Bachelor of Science - B.Sc., Industrial Engineering and Management (2019-01 - 2023-01)
- Harishonim High School — High School Diploma (2011-01 - 2014-01)

SKILLS:
Amazon Redshift, Looker (Software), Tableau, MongoDB, Microsoft SQL Server, Salesforce, Microsoft Excel, Jira, Oracle SQL Developer, Signaling System 7 (SS7), Fishtown Analytics dbt, SCCP, SQL, Python (Programming Language), Project Management, Sql server, Stitch ETL, Test Planning, ss7, Chargebee

========== Q17 — position: autofleet ==========
ID: Q17
HEADLINE: Senior Software Engineer at Apple
LOCATION: Israel

SUMMARY:
(none)

EXPERIENCE:
- Senior Software Engineer at Apple (2024-06 - present)
- Staff Software Engineer at Via (2021-11 - 2024-05)
- Software Engineering Team Lead at Via (2020-04 - 2021-11)
- Software Engineer at Via (2019-08 - 2020-04)
- Software Engineer at Gefen International AI (2018-02 - 2019-07)
- Software Engineer at Check Point Software (2015-10 - 2018-01)

EDUCATION:
- Tel Aviv University — Bachelor of Science (B.Sc.), Computer Science (2013-01 - 2017-01)
- Tel Aviv University — Bachelor of Science (B.Sc.), Computer Science (2013-01 - 2017-01)

SKILLS:
Software Development, Object Oriented Design, Agile Methodologies, Software Design, Programming, Web Development, Object-Oriented Programming (OOP), Python, Java, Node.js, React.js, JavaScript, SQL, ECMAScript, Amazon Web Services (AWS), NoSQL, MongoDB, PostgreSQL, Back-End Web Development, Redis

========== Q18 — position: autofleet ==========
ID: Q18
HEADLINE: Full Stack Developer
LOCATION: Tel Aviv-Yafo, Israel

SUMMARY:
Full stack developer. Magshimim program graduate.Eager to learn and improve.Specialize in React.js, Next.js, Node.js, and C++.

EXPERIENCE:
- Full Stack Developer at Wix (2025-09 - present)
- Full Stack Developer at Israel Defense Forces (2021-10 - 2025-01)

EDUCATION:
- Magshimim — Cyber and Computer science (2018-10 - 2021-06)

SKILLS:
Self Learning, React.js, Back-End Web Development, Front-End Development, SQLite, Next.js, TypeScript, JavaScript, SQL, SASS, HTML5, Cascading Style Sheets (CSS), Object-Oriented Programming (OOP), Sass, Scss, Gitlab, [link], C (Programming Language), C++, Python (Programming Language)

========== Q19 — position: autofleet ==========
ID: Q19
HEADLINE: Software Engineer at riseup
LOCATION: Giv'at Shmuel, Israel

SUMMARY:
(none)

EXPERIENCE:
- Software Engineer at RiseUp (2021-10 - present)
- Software Development Team Lead at DDR&D - IDF's and Ministry of Defense's R&D Unit at Israel Defense Forces (2018-09 - present)
    Managed a team of 8 software developers.
    Designed, implemented and managed full stack applications for tailor-made needs of the organization. Took part in designing project roadmaps and architectures as well as implementing them while bringing high value to end users with much concentration on UX while using various technologies like React, Node.js, Next.js,  Neo4j, MSSQL and more.
- Full Stack Developer at DDR&D - IDF's and Ministry of Defense's R&D Unit at Israel Defense Forces (2015-10 - 2018-08)
    Developed management systems tailored to the in-house needs of DDR&D of managing large scale projects and budgets.
    Developed in Angular, ASP.Net, WPF,  WCF, C#, MSSQL.

EDUCATION:
- Holon Institute of Technology — Bachelor of Science - BS, Computer Science (2012-10 - 2015-08)

SKILLS:
React.js, Node.js, C#, Next.js, Windows Communication Foundation (WCF), Neo4j, Full-Stack Development, Development Management, ASP.NET Web API, Angular, Microsoft SQL Server, Xamarin, Ionic Framework

========== Q20 — position: autofleet ==========
ID: Q20
HEADLINE: Salesforce & Full Stack Developer at Natural Intelligence
LOCATION: Israel

SUMMARY:
(none)

EXPERIENCE:
- Senior Salesforce Developer at Natural Intelligence (2023-01 - present)
- Software Engineer at BALINK (2016-05 - 2020-03)
- Software Development Team Lead at OpenApp Israel (2008-01 - 2016-05)
- Web Software Engineer at OurCrowd (2020-03 - 2022-01)

EDUCATION:
- The Jerusalem College of Technology — Bachelor of Science (B.Sc.), Computer Science

SKILLS:
Full-Stack Development, Redux.js, Node.js, React.js, MongoDB, SOQL, AWS Lambda, Amazon Web Services (AWS), Back-End Web Development, Front-End Development, Software Development, Git, Object-Oriented Programming (OOP), Linux, SQL, JavaScript, HTML, jQuery, MySQL, Java, Moodle, Salesforce.com, Microsoft SQL Server, XML, AngularJS, Team Leadership, Apex Programming, MariaDB, YUI Library

========== Q21 — position: scaleops ==========
ID: Q21
HEADLINE: Senior Staff Software Engineer at Palo Alto Networks
LOCATION: Even Yehuda, Israel

SUMMARY:
(none)

EXPERIENCE:
- Senior Staff Software Engineer at Palo Alto Networks (2022-12 - present)
- Software Developer at Cider Security (2021-08 - 2023-01)
- Python Developer at Israeli Military Intelligence - Unit 8200 (2020-05 - 2021-08)
- Cyber Analyst at Israeli Military Intelligence - Unit 8200 (2018-04 - 2020-05)

EDUCATION:
- Handasaim Herzliya High School — Mechanical Engineering

SKILLS:
REST APIs, JavaScript, Python (Programming Language), SQL, Networking, Data Analysis, Cybersecurity, Flask, SOLIDWORKS, Representational State Transfer (REST), HTML, Git, MySQL

========== Q22 — position: autofleet ==========
ID: Q22
HEADLINE: Senior Full Stack Engineer | Creator of NeatGit
LOCATION: Israel

SUMMARY:
Detail-oriented Senior Software Engineer &amp; Full Stack Developer with 10 years of experience, specializing in React &amp; Node.js. I bring strong responsibility, foresight, and a proactive approach to building high-quality applications, caring deeply about delivering value to users and the business.Experienced in designing and developing end-to-end web applications, from crafting user-friendly, responsive frontend components to implementing complex backend features that leverage databases, microservices, APIs, and modern web frameworks to deliver reliable solutions.I thrive both in the details - writing clean code, debugging, optimizing performance - and at the big picture level - planning architecture and designing systems.Always learning. Always improving.

EXPERIENCE:
- Senior Full Stack Engineer at Riverside (2025-10 - present)
    ✅ Contributed to the frontend migration from a monolith to Micro Frontends architecture, delivering the first production MF across 20 planned services.
    
    ✅ Implemented a new user entry point to a key user page, increasing monthly unique visits by 43% post-launch.
    
    ✅ Drove product experimentation using A/B testing and feature flags, enabling data-driven decision making and controlled feature rollouts.
- Full Stack Engineer at Melio (2022-02 - 2025-03)
    ✅ Developed a production-grade Risk Operations web application using Node.js and React within an AWS microservices architecture, enabling analysts to make data-driven payment approval decisions.
    
    ✅ Led the end-to-end design and development of a new user training system in the app, including writing specifications, development, deployment and testing.
    
    ✅ Constructed a complex data-driven payment prioritization system, reducing the number of delayed payments and increasing company revenue.
    
    ✅ Developed a feature that enhanced users’ insight into payment data, which helped prevent fraud and saved the company hundreds of thousands of dollars.
    
    ✅ Contributed to Melio’s Operations Platform in a Nx monorepo, creating reusable npm packages to streamline development in a shared codebase.
- Full Stack Engineer at Israeli Military Intelligence (2015-10 - 2021-07)
    ✅ Developed mission-critical web applications with React, TypeScript, and Node.js, collaborating with teams and clients to deliver software that supports military operations.
    
    ✅ Acted as co-technical lead, translating managers’ specifications into impactful, high-quality features.
    
    ✅ Proposed and implemented modern testing strategies using Jest and Enzyme to improve code quality and team efficiency.

EDUCATION:
- Ben-Gurion University of the Negev — Bachelor of Science - BS, Computer Science (2012-10 - 2015-08)

SKILLS:
(none)

========== Q23 — position: scaleops ==========
ID: Q23
HEADLINE: Senior Software Engineer at CyberArk
LOCATION: Petah Tikva, Israel

SUMMARY:
(none)

EXPERIENCE:
- Senior Software Engineer at CyberArk (2022-07 - present)
    I'm currently developing SaaS platform services for CyberArk's SaaS offerings.
- Software Developer at CyberArk (2019-08 - 2022-07)

EDUCATION:
- Bar-Ilan University — Bachelor of Applied Science - BASc, Applied Mathematics (2014-01 - 2017-01)
- InfinityLabs R&D (2019-01 - 2019-07)

SKILLS:
Cloud Applications, Serverless Computing, Amazon Web Services (AWS), C (Programming Language), C++, Shell Scripting, Git, Object-Oriented Programming (OOP), JSON, Python (Programming Language), Groovy, Jenkins, Linux, Unix, Object-oriented Software, Agile Methodologies, Bash, bash

========== Q24 — position: scaleops ==========
ID: Q24
HEADLINE: Software Engineer at Redis
LOCATION: Israel

SUMMARY:
Passionate about open-source, Linux, distributed systems, cloud-native development, and microservices architecture. Experienced in Kubernetes and Openshift.

EXPERIENCE:
- Software Engineer at Redis (2025-07 - present)
- Software Engineer at Red Hat (2022-09 - 2025-07)
    Develop and improve features for Telco customers as part of the Container Native Functions
    initiative, ensuring optimal performance for Telco 5G real-time workloads.
    Maintain and enhance OpenShift compute operators, including Cluster Node Tuning Operator and NUMA Resources Operator.
    Contribute to several open source projects.
- Software Engineer Intern at Cybereason (2022-03 - 2022-07)
    Got accepted to Starship program of Gav-Yam Negev advanced technologies park , a program designed for outstanding students at Ben Gurion University.
    
    Designed and developed secured app in client-server architecture.
    Software development in C++, Boost, Open-SSL.
- Platoon Sergeant & Combat Solider at Israel Defense Forces (2017-03 - 2019-11)

EDUCATION:
- Ben-Gurion University of the Negev — Computer Science (2020-10 - 2023-09)

SKILLS:
Operating Systems, Red Hat Enterprise Linux (RHEL), OpenShift, Software Development, Computer Science, Go (Programming Language), Docker, Kubernetes, Java, C#, C++, C (Programming Language), Python (Programming Language), Data Structures, Algorithms, Problem Solving, TypeScript, Linux

========== Q25 — position: scaleops ==========
ID: Q25
HEADLINE: Back End Tech Lead at BioCatch
LOCATION: Petah Tikva, Israel

SUMMARY:
Experienced Developer with a demonstrated history of working in software. Skilled in Python, C#, C++, Data Structures, Algorithms, and Software Development. Developed multiple Open-Source projects.

EXPERIENCE:
- Back End Technical Lead at BioCatch (2025-04 - present)
- Back End Developer at BioCatch (2020-06 - 2025-04)
- Software Engineer at Omnisys (2016-11 - 2020-04)
    Worked in Omnisys as part of IDF service

EDUCATION:
- The Open University — Master's degree, Computer Science (2018-01 - 2021-01)
- Bar-Ilan University — Bachelor's degree, Computer Science (2013-01 - 2016-01)

SKILLS:
Software, Python (Programming Language), Mathematics, Data Structures, Algorithms, C#, C++, Software Development, Programming, C (Programming Language), Programming Languages, Java, Rust, Rust (Programming Language), D, Git, Pycharm, SWIG, Geographic Information Systems (GIS), Qt

========== Q26 — position: autofleet ==========
ID: Q26
HEADLINE: Programmer
LOCATION: Israel

SUMMARY:
Software engineer with experience and drive. Especially fond of simple code and good end results.

Specialties: Learning quickly, love of programming languages, refactoring, testing, inter-language integration, code optimization, diving into legacy code, server-side, client-side web, mobile, encryption.

EXPERIENCE:
- Programmer at WekaIO (2016-06 - present)
    Filesystem internals & cluster management, filesystem encryption
- Programmer at Facebook (2013-10 - 2016-05)
    Breaking things then moving fast
- Programmer at Onavo (2011-08 - 2016-05)
    Android, iOS, and server-side development. Technical leader.
- Project Lead at Testoob (2005-05 - 2011-06)
    Project lead on Testoob, an advanced open-source testing framework for Python.
    [link]
- Programmer at CloudShare (2008-01 - 2010-11)
    Server-side and client-side WEB with .NET and JS
- Programmer at N/A (2004-08 - 2008-01)
    Working with several teams in different languages and environments, developing software and mentoring other programmers. At one time engineered the rewrite of a large C++ application core in Python, relying on the extensive testing we put in place. The project finished earlier than expected and the end results were a significantly simpler and more maintainable code base and a 4-fold to 70-fold speed increase.
- MIS (system) at HumanEyes (2002-05 - 2004-03)
    Was responsible for the company's IS and IT needs. Managed Mac and Windows workstations and FreeBSD servers. Worked part time.
- Programmer at N/A (1997-08 - 2001-10)
    Software developer, worked extensively with C++, C, Python, and Java. Started as a trainee for 6 months. Worked in 2 different teams.

EDUCATION:
- The Hebrew University of Jerusalem — BSC, Mathematics and Computer Science (2001-01 - 2009-01)

SKILLS:
Python, Software Development, C#, Linux, C++, Java, Software Design, Ruby, Object Oriented Design, Nice Guy, C, .NET, Agile Methodologies, OOP, JavaScript, Unix, Multithreading, Android, Web Development, Client Side

========== Q27 — position: scaleops ==========
ID: Q27
HEADLINE: Software Developer
LOCATION: Israel

SUMMARY:
(none)

EXPERIENCE:
- Software Engineer at Stealth Startup (2025-01 - present)
- Software Developer at Intel GmbH (2021-09 - 2024-07)

EDUCATION:
- Technion - Israel Institute of Technology — Bachelor of Science - BS, Computer Science (2020-01 - 2024-01)

SKILLS:
Python (Programming Language), Machine Learning, Natural Language Processing (NLP), C++, C (Programming Language), Project Management, Team Leadership, Jira, Data Analysis, Business Strategy, Analytical Skills, Creative Strategy, Communication, Management, English, R (Programming Language), Teamwork, Problem Solving, Data Structures, Algorithms

========== Q28 — position: scaleops ==========
ID: Q28
HEADLINE: R&D Team Lead at Rivery
LOCATION: Tel Aviv District, Israel

SUMMARY:
(none)

EXPERIENCE:
- R&D Team Lead at Rivery (2022-12 - present)
- Senior Back End Developer at Explorium (2021-08 - 2022-08)
- Back End Developer at WSC Sports (2017-07 - 2021-08)
- Senior Backend Engineer at Rivery (2022-08 - 2022-12)

EDUCATION:
- Ben-Gurion University of the Negev — Bachelor Science (BSc), Computer Science (2013-01 - 2016-01)

SKILLS:
Python, C#, Java, SQL, MongoDB, Amazon Web Services (AWS), Microsoft Azure, REST APIs, Kubernetes, Redis, SQLAlchemy, Celery, RabbitMQ, Data Structures, C++, C, Git, Windows, Linux

========== Q29 — position: autofleet ==========
ID: Q29
HEADLINE: Senior Software Engineer | Microsoft
LOCATION: Modi'in-Maccabim-Re'ut, Israel

SUMMARY:
(none)

EXPERIENCE:
- Senior Software Engineer at Microsoft (2022-04 - present)
- Senior Data & Backend Engineer at Moovit (2017-05 - 2022-04)
- Backend & Big Data Engineer at Viber Media, Inc. (2013-07 - 2017-05)
- Team Leader at Ex Libris (2007-07 - 2013-07)
- Team Leader at Radware (2005-12 - 2007-07)
- Programmer at GMOD (2002-01 - 2005-01)

EDUCATION:
- Bar-Ilan University (1993-01 - 1996-01)

SKILLS:
XML, OOP, Java Enterprise Edition, Object Oriented Design, Multithreading, Databases, Software Development, System Architecture, Unix, Java, Design Patterns, Agile Methodologies, SQL, Software Design, Linux, Eclipse, Management, SaaS

========== Q30 — position: scaleops ==========
ID: Q30
HEADLINE: Software Engineer at Meta
LOCATION: Israel

SUMMARY:
(none)

EXPERIENCE:
- Software Engineer at Meta (2021-12 - present)
- Instructor at QueenB (2020-10 - 2021-12)
- Software Engineer Intern at Facebook (2020-07 - 2020-10)
- Veterinary Technician at Ramat Hasharon Veterinary Center (2017-10 - 2020-07)

EDUCATION:
- Bar-Ilan University — computer science and neuroscience, computer science and neuroscience (2018-01 - 2021-01)
- Bar-Ilan University — computer science and neuroscience, computer science and neuroscience (2018-01 - 2021-01)

SKILLS:
c, java, Cell Biology, Organic Chemistry, Discrete Mathematics, C++, C#, Python (Programming Language), Reinforcement Learning, Linux

========== Q31 — position: autofleet ==========
ID: Q31
HEADLINE: Senior Backend Engineer at Slice Global
LOCATION: Israel

SUMMARY:
Experienced Software Engineer with a demonstrated history of working in the consumer goods industry. Skilled in Object Oriented Design, Node.js, SQL and non SQL databases. Strong engineering professional graduated from The Hebrew University. Good team player.

EXPERIENCE:
- Senior Backend Developer at Slice | Global Equity (2025-06 - present)
- Software Development Engineer at BabyRoo (2018-06 - 2018-12)
    Half a year project in a startup in Munich, Germany. Code development from scratches to create MVP up and running until the project deadline.
- Full Stack Engineer at Dynamic Yield (2019-03 - 2022-08)
- Senior Backend Developer at Metis (2022-08 - 2025-06)
- C++ Software Developer at SintecMedia (2016-03 - 2016-10)

EDUCATION:
- The Hebrew University — Computer Engineering (2013-01 - 2016-01)

SKILLS:
OpenTelemetry, NoSQL, NestJS, Node.js, TypeScript, Full-Stack Development, OOP, Software Development, Programming, Version Control, Python, JavaScript, SQL, Amazon Web Services (AWS), Elasticsearch, docker, Interpersonal Skills, Back-End Web Development, DBMS, Redis, Kubernetes

========== Q32 — position: scaleops ==========
ID: Q32
HEADLINE: AI/ML Security Researcher at Cisco | Co-Lead, OWASP Securing Agentic Applications | PhD in AI/Cybersecurity | Advancing Agentic AI Safety
LOCATION: Tel Aviv-Yafo, Israel

SUMMARY:
With a Ph.D. in Software and Information Systems Engineering from Ben-Gurion University, I specialize in transforming complex AI security theory into practical, actionable guidance. As a Co-Lead of the OWASP Securing Agentic Applications project, I am at the forefront of this effort, architecting industry-wide security standards like OWASP Agentic TOP 10, Agent Name Service (ANS) and the Agent-to-Agent Secure (A2AS) protocol to ensure safe and reliable agentic ecosystems.
My background in the IDF's elite "Psagot" program provided a foundation for leadership and technical execution. Today, my work translates directly into value for the security community and the enterprise, with a patent and multiple publications in top-tier venues such as IEEE TAES, CARS, CVPR, and ACM CSUR. These contributions provide organizations with concrete frameworks and strategies to defend against emerging threats. I am passionate not just about building secure intelligent systems, but about providing the clear, actionable guidance necessary for their widespread and safe adoption.

EXPERIENCE:
- Core Team - OWASP Agentic Security Initiative at OWASP GenAI Security Project (2025-01 - present)
    Co-Leading Securing Agentic Applications Workstream
    Core Contributor - TOP10 for Agentic AI
- Senior Technical Lead - AI Security Researcher at Cisco (2025-12 - present)
    Driving Secure AI Strategy, Research & Technical Execution
- Founding Member at OWASP AIVSS Project (2025-05 - present)
- AI & Security Researcher at CBG - Cyber at Ben Gurion University (2019-10 - present)
- Staff AI Security Researcher at Intuit (2024-03 - 2025-10)
- Site Reliability Engineering Group Leader at Israeli Military Intelligence - Unit 8200 (2022-08 - 2024-03)
- Security Researcher at Israeli Military Intelligence - Unit 8200 (2017-07 - 2020-10)
- Team Leader at Israeli Military Intelligence - Unit 8200 (2020-02 - 2023-04)
    Leading a team consits of 6 security researchers, involved with multiple projects in parallel. Low level Research, red teaming, network analysis, security tools development.

EDUCATION:
- Ben-Gurion University of the Negev — Bachelor of Science - BS, Computer Software Engineering (2013-10 - 2017-10)
- Ben-Gurion University of the Negev — Doctor of Philosophy - PhD, Computer Software Engineering (2019-10 - 2024-04)
- Ben-Gurion University of the Negev — Master of Science - MS, Computer Software Engineering (2016-10 - 2018-10)
- Psagot — Class 15 (2013-10)

SKILLS:
Threat Modeling, AI Security, AI Safety, Agentic AI, Adversarial AI, Data Science, Cyber Risk Management, Anomaly Detection, Deep Learning, Site Reliability Engineering, Data Analysis, Network Operations Center (NOC), Security Operations Center, Avionics, Software Development, Machine Learning, Cybersecurity, Research, Network Security

========== Q33 — position: autofleet ==========
ID: Q33
HEADLINE: Staff Software Engineer at Google
LOCATION: Israel

SUMMARY:
Specialties:
Objective-C // Swift // C++ // Java
Postgres // MySQL // MongoDB // IndexedDB
Node.js // PHP // JavaScript (jQuery, AngularJS) // HTML5 // CSS3
WordPress // Drupal // Kirby
Matlab // Octave // OCaml

Adobe Photoshop, Premiere Pro, Microsoft Office Suite (Windows & Mac), UPS WorldShip, QuickBooks Pro

macOS // Linux // Windows

EXPERIENCE:
- Staff Software Engineer at Google (2025-04 - present)
- Senior Software Engineer, Waze at Google (2021-04 - 2025-10)
- Senior Software Engineer, Waze at Google (2017-11 - 2021-04)
- Software Engineer, Search / Maps at Google (2017-02 - 2017-11)
- Consulting Engineer at IQTELL.COM LLC (2015-09 - 2017-08)
    Continued to improve the IQTell iOS app on a consulting basis and contributed to development of a new React web app.
    Responsible for maintaining the public marketing site as well as the Knowledge Base help site.
- Software Engineer, Zagat at Google (2015-08 - 2017-02)
- Lead iOS Engineer at IQTELL.COM LLC (2014-03 - 2015-08)
    Leading design and development of the highly advanced Email and Task management app for iPhone and iPad.
    
    Responsible for implementing new functionality, maintaining the code base following industry best practices, managing feedback, interacting with team members and customers to perfect the app, and releasing the app to beta users and the App Store.
    
    Working with:
    - UIKit
    - EventKit
    - AddressBook
    - Core Graphics
    - iCloud KV for cross-device consistency
    - Apple Push Notification Service for realtime updates
    - Core Data to make it all work offline
    
    all communicating with a RESTful JSON web service to sync data to the web and other devices.
    
    
    Also maintaining my role as Lead Developer for the IQTELL EZ Bar Chrome Extension as well as the IQTELL Knowledge Base help site.
- Web & iOS Development, Support & Quality Assurance at IQTELL.COM LLC (2011-07 - 2014-02)
    * Lead Developer for the IQTELL EZ Bar, a Chrome Extension built on IndexedDb.
    * Created the IQTELL Knowledge Base, designed as a self-help site for customers, featuring search, responsive design and an administrative panel for content creation.
    * Develop the IQTELL iOS app using a RESTful API on top of Core Data and heavily customized User Interface.
    * Responsible for Support and Quality Assurance for a promising start-up
    * Created an automated report for user activity statistics, business intelligence and targeted email marketing.
    * Advise in future planning, interface design, user experience and perform market research.
    * User acquisition through social media and marketing, grew user base by 30% shortly after initially joining the company.
- IT at  (2007-02 - 2014-01)
    - Organized and effectively transitioned Oxygen Imports to Google Apps, cutting IT costs while increasing employee productivity.
    - Computer network system setup and maintenance, hardware and software upgrades, terminal configuration, end-user support.
    - Web site management ([link] design of catalog sheets and business cards, product photography, preparation of Press Releases, assist in Trade Shows and showroom setup, receiving and shipping of goods in warehouse, unloading containers.
    - Order entry, invoicing using QuickBooks accounting program, credit card processing
- Help Desk Consultant at Rutgers University (2010-09 - 2011-09)
    - Provided high-quality technical assistance to the Rutgers New Brunswick community, documenting and resolving user calls using a detailed ticketing system.
    - Received award for Customer Service Appreciation, March 2011
- Training Department at Vericle - Your Medical Billing, Coaching, and Compliance Business: Software, Staff, and Operations (2009-07 - 2010-08)
    Help build and develop a learning center for medical billing software. Design custom graphic and video tutorials (scripting, voice recording and screen capturing) and quizzes to train medical office employees.
- Sales at  (2004-01 - 2008-01)
    Order entry, order shipment, billing and invoicing, credit card processing.

EDUCATION:
- Marlboro High School — Center for Business Administration, Center for Business Administration (2006-01 - 2010-01)
- Rutgers University — BS, Computer Science (2010-01 - 2013-01)
- Rutgers, The State University of New Jersey-New Brunswick — BS, Computer Science (2010-01 - 2013-01)
- Marlboro High School — Center for Business Administration (2006-01 - 2010-01)

SKILLS:
iOS, iOS Development, Swift (Programming Language), Mobile Applications, HTML, Microsoft Office, Photoshop, Matlab, Java, OCaml, CSS, Quickbooks, Hebrew, JavaScript, Customer Service, Word, Windows, PowerPoint, Excel, Mac OS X

========== Q34 — position: autofleet ==========
ID: Q34
HEADLINE: Freelance Full Stack Developer && Programmer
LOCATION: Israel

SUMMARY:
Full Stack Developer. Creating ideas into production ready applications. Web applications, Mobile Applications for IOS and Android, and many other software solutions

EXPERIENCE:
- Full Stack Developer at Freelance Web Development (2017-05 - present)
- Full Stack Developer at RecCenter (2016-11 - 2017-05)
- Software Engineer at Aqua Security (2015-01 - 2016-11)

EDUCATION:
- Ben-Gurion University of the Negev — Bachelor of Arts (B.A.), Political Science and Economics (2003-01 - 2006-01)
- Sela Collage — Cyber-Security, Networking, Computer and Information System security (2013-01 - 2014-01)

SKILLS:
Mobile Applications, React.js, Go, Android Development, iOS Development, Web Development, Programming, Software Development, Object Oriented Design, Game Development, Software Design, Web Applications, Design Patterns, Swift, iOS, React Native, Android, Java, Next.js, GraphQL

========== Q35 — position: scaleops ==========
ID: Q35
HEADLINE: Full stack software developer
LOCATION: Israel

SUMMARY:
Graduated software developer (B.Sc in computer science), specilized in object-oriented…

EXPERIENCE:
- Software Developer at Tipalti (2021-07 - present)
- Producer at ERM Advanced Telematics (2015-10 - 2016-10)

EDUCATION:
- The Academic College of Tel-Aviv, Yaffo — Bachelor of Science - BS, Computer Science (2016-01 - 2019-01)

SKILLS:
c#, C++, SQL, C (Programming Language), Design Patterns, Linux, Java, Software Architecture, x86 Assembly, Computer Networking, Object-Oriented Programming (OOP)

========== Q36 — position: autofleet ==========
ID: Q36
HEADLINE: Full Stack Developer at ZoomInfo
LOCATION: Israel

SUMMARY:
Enthusiastic and highly motivated FullStack Developer.

Specialized in Angular, NodeJS, Python, .NET

Computer Science graduate - Holon Institution of Technology.

In addition with a Practical Engineer degree focused in Electronics &  Computer Engineering.

EXPERIENCE:
- Full Stack Developer at ZoomInfo (2022-07 - present)
- QA Automation Engineer at IAI - Israel Aerospace Industries (2018-07 - 2020-06)
- Full Stack Developer at ONE Digital (2021-05 - 2022-07)
- Electronic Practical Engineer at IAF Israel Air Force (2012-06 - 2015-06)
- Integration Engineer at HP (2015-06 - 2018-05)
- Process Developer at IAI - Israel Aerospace Industries (2020-05 - 2021-06)

EDUCATION:
- ORT Colleges — Electronic practical engineer (2010-01 - 2012-01)
- Ort Afridar Ashkelon — Ort Rehovot College, Electrical, Electronics and Communications Engineering (2006-01 - 2010-01)
- Holon Institute of Technology — Computer science (2017-01 - 2020-01)

SKILLS:
NestJS, Angular, Android Development, Web Development, Software Development, Integration, Electronics, Testing, Quality Assurance, Python (Programming Language), C++, C, SAP Products, C (Programming Language), Microsoft Office, C#, Java, SQL, JavaScript, Visual Studio, Android, MySQL, javascript, Oracle SQL Developer, English, Android Studio, MVC, Node.js, MongoDB, [link], Bootstrap, Spring Boot, React, Redux

========== Q37 — position: autofleet ==========
ID: Q37
HEADLINE: Software Architect at Verifind
LOCATION: Israel

SUMMARY:
Innovative technology leader with over 8 years of experience architecting and delivering complex, scalable web applications. Specializing in Node.js, React, cloud-native architectures, and modern CI/CD practices, I excel at transforming engineering teams and optimizing digital solutions to drive business growth and superior user engagement.

Core Skills:
* Software Development: JavaScript, TypeScript, Node.js, React.js, Express.js
* Cloud & DevOps: AWS, Azure, Google Cloud, Docker, Kubernetes, CI/CD pipelines
* System Architecture: Microservices, serverless architectures, high-availability systems
* Team Leadership: Mentoring, cross‑functional collaboration, agile project management

EXPERIENCE:
- Software Architect at Verifind (2024-04 - present)
- Full-Stack Developer at Agently.co (2018-08 - 2019-07)
    (Node.js, Express.js, Vanilla JS, Firebase)
    
    Agently is a software company that provides a digital platform for real estate agents to communicate comprehensive information about their assets.
    
    As the only programmer in the startup, I:
    
    ◦ Created the entire platform from scratch
    
    ◦ Wrote entire server & client sides
    
    ◦ Integrated databases, authentications, storage, analytics and more (mostly via Firebase)
- React.js Developer at Just Eat Takeaway.com (2019-08 - 2021-09)
    (React.js, Node.js, Next.js, SEO, Azure Cloud, Azure Pipelines)
    
    TakeawayPay is a project of the worldwide Just Eat Takeaway.com (which lately aqquired 10bis.co.il for 157 million dollars). TakeawayPay is a unique innovation in the food delivery industry. The platform enables companies to register online and issue food ordering allowances to their employees. This very exciting project grew into a worldwide sensation quickly, serving 13 (soon 16) countries.
    
    ◦ Worked at the forefront of development, from scratch, of Takeaway Pay (including 13+ localized websites, for example [link] using the latest web technologies, and the further development of the website.
    
    ◦ Guided integrations on the website- improved the traffic to the website from Google, working with the SEO and analytics teams.
    
    ◦ Was responsible for the infrastructure of the project- 3rd party packages, Webpack, Babel, Azure Pipelines.
    
    ◦ Reported to the project manager & introduced demos to the investors.
- Full-Stack Developer at Welldone Software Solutions Ltd. (2019-08 - 2021-09)
    (JavaScript, Node.js, MongoDB, Angular, React.js, Sencha, Cordova, Android, iOS, Windows Phone, PHP, SQL)
    
    A boutique software solutions company is known as a company that fosters a culture of excellence and implements the best software solutions for the companies that it works with.
    
    Worked on a variety of start-ups, developing front-ends, back-ends, databases, and mobile applications.
    
    ◦ Served as a source of knowledge in various projects.
    
    ◦ Worked on various projects in new and diverse technologies.
    
    ◦ Published NPM packages / maintained VSCode extensions on behalf of the company.
    
    ◦ Worked directly with the CEO.
- Full-Stack Developer at Freelancer (2017-08 - 2018-07)
    As a freelance programmer, while studying, I took on personal projects for a variety of businesses and dozens of satisfied clients..
    
    Front-End / Any front-end request for mobile-apps, websites and web-apps. Projects included for Android, iOS, Webflow, CSS, jQuery, Angular, React and more.
    
    Back-End / Server side programming. Projects included code in Java, C, C#, C++, Python, Node.js, Express.js and more.
- Tech Lead at Vi (2021-09 - 2025-05)
    (Node, React, Microservices Architecture, Cloud Architecture (AWS & GCP), CI/CD Pipelines, Containers, SQL & NoSQL Databases)
    
    Vi builds enterprise SaaS products that help organizations improve engagement, communication, and acquisition processes. I led the end-to-end technical development of Vi Engage and Vi Acquire from the ground up, working across architecture, backend, frontend, and DevOps while coordinating multiple teams.
    
    • Led technical direction and development for Vi Engage and Vi Acquire, defining architecture, workflows, and delivery processes.
    • Managed and mentored a team of 8+ developers, establishing coding standards, code review practices, and development guidelines.
    • Designed microservices-based architectures supporting over 1M monthly active users.
    • Implemented REST and GraphQL APIs, caching layers, event-driven workflows, and real-time communication.
    • Built CI/CD pipelines ([link] Actions, AWS CodeDeploy) enabling zero-downtime deployments.
    • Deployed cloud solutions using AWS (Lambda, EC2, S3, CloudFront, DynamoDB, RDS, Elasticache).
    • Reduced infrastructure costs by 35% through serverless patterns and performance optimizations.
    • Created automated testing environments (Jest, Cypress) and monitoring/alerting systems for reliability.
    • Delivered full-stack features across React.js, Next.js, TypeScript, Node.js, and Express.js.
    • Built internal frameworks, shared UI libraries, data models, analytics flows, and permission systems.
    • Led the first production releases of Vi Engage and Vi Acquire, including architecture, development, and deployment.
    • Improved development velocity by 25% through architectural restructuring and optimized engineering processes.
    • Collaborated with product, design, and customer teams to ensure fast and scalable feature delivery.

EDUCATION:
- Technion - Israel Institute of Technology — Associate's degree, TCMP & Software Architecture (2014-01 - 2015-01)
- Reichman University — Bachelor of Science (BSc), Computer Science (2015-01 - 2018-01)

SKILLS:
TypeScript, Management, Leadership, Entrepreneurship, Software Development, JavaScript, C (Programming Language), C#, Node.js, C, C++, Python, jQuery, Express.js, Firebase, React.js

========== Q38 — position: autofleet ==========
ID: Q38
HEADLINE: Java Backend Developer at BIT
LOCATION: Israel

SUMMARY:
[link] Page : [link]
Personal web: [link]
Trelix Project : [link]
Uleadz Project : [link]

EXPERIENCE:
- Full Stack / Java Developer at Bank Hapoalim בנק הפועלים (2022-02 - present)
    As part of the position:
    ● Developed back-end logic in Java to implement a bank platform that use to give
        service to all bank branches.- Java, Spring boot, Maven, Gradle, Apigee, Helm,
      Docker, API, Mq, Rabbit, CI CD, Splunk.
    ● Develop Restful APIs and Micro-services for bank platform - Artifactory, Jenkins,
     Mongodb, JSP, SQL, MYSQL, Linux, Hibranate.
- Java Backend Developer at LADPC (2024-10 - present)
    As part of the position:
    ● Java, Spring boot, kafka, OCP, Docker, MongoDB,
    ● SQL, Apigee. GIT . control m, CI CD, Jira
- IRA Investment Group at Meitav Dash (2020-02 - 2021-06)
    IRA Investment Group.
    ● Alternative Investments Desk
    ● Management of funds and investments
    ● Communication with companies, banks and investors.
- Full Stack Developer at Uleadz (2021-09 - 2022-02)
    As part of the position:
    ● Java, JavaScript, Net, Node.js, C#, GIT,
    ● MySQL, MongoDB. web server . and UI/UX/css/scss design
- Finance and Capital Markets at Bank Leumi (2018-02 - 2020-02)
    As part of the position:
    ● Operation of Israeli mutual funds
    ● Investment operations - updating securities data in Israel and abroad 
    ● Investment revaluation and performance analysis, production of management reports
- Operations Coordinator - Provident funds at Altshuler Shaham (2017-01 - 2018-02)
    As part of the position:
    - Assist in finalizing investment management agreements
    - Identify and research potential Fund Managers; maintain and develop formal manager research documentation

EDUCATION:
- The Open University of Israel — student B.Sc computer science (2022-01)
- Ono Academic College — B.A. Business Administration, & Economics (2017-01 - 2020-10)
- Coding Academy Israel — Full Stack Developer (2021-07 - 2021-12)

SKILLS:
Spring Boot, Wso2, oauth2, tcs, Eclipse, swager, JUnit, WinSCP, Splunk, Apigee API Management, Scss, HTML, Uiux, Microsoft SQL Server, Cypress, Java, Sonar, Jenkins, Bitbucket, Jira

========== Q39 — position: autofleet ==========
ID: Q39
HEADLINE: Senior Software Engineer at Taboola
LOCATION: Israel

SUMMARY:
Experienced Software Engineer with a demonstrated history of working in the cloud…

EXPERIENCE:
- Senior Software Engineer at Taboola (2021-03 - present)
-  at Microsoft ( - present)
- Software Engineer at Digital Equipment Corporation (2015-01 - 2019-03)
    Responsibilities: 
    •	Developing tools  for multi datacenter public/private cloud environment:
    o	Server side: JAVA, Maven, Spring, GIT, Apache Tomcat ,Jenkins, coding on AWS, both for Linux and windows servers, multithreaded tools, writing scripts using Python 2.7, writing some C/C++ code for Linux machines 
    o	SQL – Oracle/MS-SQL/Postgres, building queries against product DB and creating new structures for our own tools
    o	DevOps Tools: setting up and managing Jenkins, Docker, designing architecture in AWS and Terraform 
    o	 Client side: HTML, JavaScript, Angular, jQuery, Ajax React 
    •	Planning, Automating and executing major operations and changes, such as big volumes of data migration, datacenter migration, migration to AWS, integration of new software and frameworks   
    •	Managing our students team : recruiting, interviewing, training and managing daily work of our students (4-6 students)
    •	Managing different ongoing projects: Identifying the requirements and challenges, representing the team in meetings with other Stakeholder, designing the solution and leading the task force to resolution 
    •	Latest projects I finished is : Migration of production private cloud data center in Sydney to AWS (400 servers and 100 enterprise customers) - Working on new architecture in AWS, Designing the process of migration, leading team of 7 engineers to develop needed tools and scripts, managing and coordinating the execution   
    •	administrating QC/PC/ALM application including creating automations for installation, upgrading, configuration in cloud environment
    •	providing last tear support for incidents and issues to other engineers 
    •	providing trainings to other engineers on new technologies: was responsible for training and qualifying new team we set up in Bulgaria from zero, training new engineers and students
    •	Reviewing and approving: code, tools design for other engineers

EDUCATION:
- mekif g — High School, Computer Scienceg (1998-01 - 2003-01)
- Hamikhlalah Ha'academit Lehandassah Sami Shamoon — BS, Computer Software Engineering (2013-01 - 2016-01)

SKILLS:
Programming, Software Engineering, SQL, Java, HTML, JavaScript, Software Development, Databases, Process Improvement, Agile Methodologies, C++, XML, C#, Microsoft SQL Server, Testing, C, Training, Team Leadership, Web Services, Python, CSS, Operating Systems, Linux, AWS, Git, Bash, Teamwork, Object-Oriented Programming (OOP), Object Oriented Design, ALM, HP Quality Center, HP Performance Center, Oracle Database Administration, Teaching, Leadership, Management, Project Management, Software as a Service (SaaS), aws, saas

========== Q40 — position: scaleops ==========
ID: Q40
HEADLINE: Associate Software Engineer at CyberArk | B.Sc in Computer Science
LOCATION: Israel

SUMMARY:
A motivated Software Developer with a B.Sc. in Computer Science and a strong passion for technology, problem-solving, and continuous learning.

During my military service in Unit 8200, I worked with complex technological systems in high-pressure environments, developing strong analytical thinking, responsibility, and teamwork skills.

I enjoy building software, learning new technologies, and working with people. I’m a team player, highly motivated, and always eager to grow and take on new challenges.

Technologies I’ve worked with:
Java, C#, JavaScript, React, Node.js, .NET, SQL, MongoDB, C++.

EXPERIENCE:
- Associate Software Engineer at Palo Alto Networks (2026-03 - present)
- Associate Software Engineer at CyberArk (2025-10 - present)
- CyberArk Bootcamp at CyberArk (2025-09 - 2025-10)
- Application Engineer at AU10TIX (2019-11 - 2021-10)
    Responsible for defining and integrating new data into the product according to customer requirements, as well as Collaboration in configuring and optimizing Machine Learning tools in Image Processing. Collaborated with QA and development teams to improve the product.
- Mobile communication system operator at Israeli Military Intelligence - Unit 8200 (2016-08 - 2019-03)
    - Managed and troubleshot classified systems across multiple applications, ensuring high reliability and 
    availability. 
    - Conducted data analysis for major intelligence operations, contributing to mission success with 
    advanced communication protocols. 
    - Led a team in a high-stress environment, optimizing task execution and collaboration.

EDUCATION:
- Zero To Mastery Academy — python
- The Academic College of Tel-Aviv, Yaffo — Bachelor's degree, Computer Science (2021-11 - 2025-03)
- Zero To Mastery Academy — python
- The Academic College of Tel-Aviv, Yaffo — Bachelor's degree, Computer Science (2021-11 - 2025-03)

SKILLS:
Python, Aws, Node.js, Docker, React.js, Java, C#, Spring Boot, JavaScript, .NET Framework, HTML, Cascading Style Sheets (CSS), Object-Oriented Programming (OOP), Design Patterns, [link], PostgreSQL, MongoDB, Communication Systems, Attention to Detail, Teamwork

