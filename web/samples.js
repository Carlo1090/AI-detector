"use strict";

// Example texts for the "Try a sample" buttons.
// - ai: written by an AI model in typical assistant style.
// - human: opening of a human-written student essay from the Ghostbuster dataset (Verma et al., 2024;
//   CC BY 3.0, https://github.com/vivek3141/ghostbuster-data), essay test split, id 6698.
// - mixed: the same human essay with one AI-written paragraph inserted in the middle.

const HUMAN_P1 = "Theodore Roosevelt was one of the youngest American presidents known for his progressive ideas and the desire to change society, addressing the existing distrust and helplessness. The decision to promote progressive ideas was not spontaneous, and he saw the movement as the possibility to protect the real rule of humans (Roosevelt, 1912). The characteristics of a progressive are the intention to stand for social justice, achieve good for all people, and improve the environment in which individuals should grow and develop. Being a progressive means being optimistic about every initiative and participating in problem-solving processes. There is no need to hide behind or keep silent but to act and become passionate about changes and transformations.";

const HUMAN_P2 = "Despite the desire to support progressive thoughts, there are also some examples of anti-progressives. Roosevelt (1912) called them the people with narrow vision and little sympathy. Anti-progressives are never stirred by the mistakes or failures made by other people (Roosevelt, 1912). Individuals who are consumed by their selfishness are not able to appeal to conscience. Everything they can do is to support oppression and use reforms that work for the leaders’ success, not the public one. The activities varied from employees’ exploitation to the promotion of capitalism that bounded society and imposed limitations on creativity and progress.";

const AI_PARAGRAPH = "Moreover, Roosevelt's progressive vision played a pivotal role in shaping the modern American state. By championing landmark initiatives such as trust-busting, consumer protection, and the conservation of natural resources, he laid the groundwork for a more equitable society. His Square Deal underscored the importance of balancing the interests of business, labor, and the public, fostering a sense of shared responsibility that continues to resonate today. Ultimately, Roosevelt demonstrated that bold leadership, grounded in a commitment to the common good, can transform both institutions and public expectations.";

const SAMPLES = {
  ai: `In today's fast-paced digital world, social media plays a pivotal role in shaping how students learn, communicate, and build relationships. Platforms like Instagram, TikTok, and X have transformed the educational landscape, offering both remarkable opportunities and significant challenges.

On one hand, social media fosters collaboration and provides access to a wide range of educational resources. Students can join study groups, follow experts in their field, and engage with diverse perspectives from around the globe. Moreover, these platforms can serve as a powerful tool for creativity, allowing learners to showcase their work and receive valuable feedback.

On the other hand, it is important to note that excessive use can have a profound impact on academic performance and mental well-being. Constant notifications disrupt focus, while the pressure to maintain a curated online presence can lead to anxiety and low self-esteem. Furthermore, the spread of misinformation underscores the need for strong digital literacy skills.

Ultimately, the key lies in striking a balance. By setting healthy boundaries, practicing mindful consumption, and leveraging social media as a learning tool rather than a distraction, students can harness its benefits while minimizing its drawbacks. Schools and parents also play a crucial role in guiding young people as they navigate the complexities of the digital age. In conclusion, social media is not inherently good or bad — it is a tool whose impact depends on how we choose to use it.`,
  human: `${HUMAN_P1}\n\n${HUMAN_P2}`,
  mixed: `${HUMAN_P1}\n\n${AI_PARAGRAPH}\n\n${HUMAN_P2}`,
};
